from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

from baseline import model as make_baseline
from kairos import features
from kairos.data import load_followers
from kairos.label import author_level
from kairos.metrics import score
import pandas as pd
import numpy as np
import json

HOOK_CACHE = Path(
    '/Users/ekaterina/kairos-dataset/results/embeddings/hooks_e5_small.npz'
)


def load_hook_embeddings(cache_path):
    with np.load(cache_path, allow_pickle=False) as saved:
        authors = saved['author']
        posts = saved['post']
        vectors = saved['vectors']
        has_text = saved['has_text']


    columns = [f'hook_e5_{number}' for number in range(384)]
    result = pd.DataFrame(vectors, columns=columns)
    result['author'] = authors
    result['post'] = posts
    result['hook_has_text'] = has_text.astype(int)
    columns.append('hook_has_text')
    return result, columns


def load_repost_features(data_dir, post_keys):
    required = set(zip(post_keys["author"], post_keys["post"]))
    rows = []

    for author in sorted(post_keys["author"].unique()):
        path = data_dir / "posts" / f"{author}.json"
        with path.open(encoding="utf-8") as source:
            feed = json.load(source)

        for post in feed["posts"]:
            key = (author, post["urn"])
            if key not in required:
                continue

            shared = post.get("resharedPost")
            rows.append({
                "author": author,
                "post": post["urn"],
                "reposted_flag": int(post.get("reposted") is True),
                "has_reshared_post": int(isinstance(shared, dict) and bool(shared)),
            })


    return pd.DataFrame(rows, columns=[
        "author", "post", "reposted_flag", "has_reshared_post",
    ])


def recent_history_features(posts):
    ordered = posts.sort_values(
        ["author", "posted_at"], kind="mergesort"
    )
    result = ordered[["author", "post"]].copy()
    result["recent_last_resid"] = np.nan
    result["recent_five_mean_resid"] = np.nan

    for _, group in ordered.groupby("author", sort=False):
        for post in group.itertuples():
            past = group.loc[
                group["posted_at"].lt(post.posted_at), "resid"
            ]
            if past.empty:
                continue

            result.loc[post.Index, "recent_last_resid"] = past.iloc[-1]
            result.loc[post.Index, "recent_five_mean_resid"] = (
                past.tail(5).mean()
            )

    return result


def load_profile_texts(data_dir):
    rows = []

    for path in sorted((data_dir / "authors").glob("AUTHOR_*.json")):
        with path.open(encoding="utf-8") as source:
            profile = json.load(source)

        parts = [
            profile.get("headline") or "",
            profile.get("summary") or "",
        ]

        for skill in profile.get("skills") or []:
            parts.append(skill.get("name") or "")

        for position in profile.get("fullPositions") or []:
            parts.append(position.get("title") or "")

        rows.append({
            "author": profile.get("author", path.stem),
            "profile_text": "\n".join(parts),
        })

    return pd.DataFrame(rows)

def run(train, validation, posts, data_dir):
    hook_features, hook_columns = load_hook_embeddings(HOOK_CACHE)
    print('Hook embedding rows:', len(hook_features))


    post_keys = pd.concat([
        train[["author", "post"]],
        validation[["author", "post"]],
    ], ignore_index=True)
    repost_features = load_repost_features(data_dir, post_keys)
    repost_columns = ["reposted_flag", "has_reshared_post"]

    profiles = load_profile_texts(data_dir)

    train_profiles = profiles[
        profiles["author"].isin(train["author"])
    ].copy()

    val_profiles = profiles[
        profiles["author"].isin(validation["author"])
    ].copy()

    profile_vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    profile_train_tfidf = profile_vectorizer.fit_transform(
        train_profiles["profile_text"]
    )
    profile_val_tfidf = profile_vectorizer.transform(
        val_profiles["profile_text"]
    )

    profile_svd = TruncatedSVD(
        n_components=50,
        n_iter=5,
        random_state=42,
    )

    profile_train_vectors = profile_svd.fit_transform(
        profile_train_tfidf
    )
    profile_val_vectors = profile_svd.transform(
        profile_val_tfidf
    )

    profile_columns = [
        f"profile_svd_{number}" for number in range(50)
    ]

    profile_train_features = pd.DataFrame(
        profile_train_vectors,
        columns=profile_columns,
    )
    profile_train_features["author"] = (
        train_profiles["author"].to_numpy()
    )

    profile_val_features = pd.DataFrame(
        profile_val_vectors,
        columns=profile_columns,
    )
    profile_val_features["author"] = (
        val_profiles["author"].to_numpy()
    )

    profile_features = pd.concat(
        [profile_train_features, profile_val_features],
        ignore_index=True,
    )


    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    text_train = vectorizer.fit_transform(train["text"])
    text_val = vectorizer.transform(validation["text"])

    svd = TruncatedSVD(
        n_components=100,
        n_iter=5,
        random_state=42,
    )

    compressed_train = svd.fit_transform(text_train)
    compressed_val = svd.transform(text_val)
    print('Post text features:', compressed_train.shape)

    levelled = author_level(posts)

    selected_authors = set(train["author"]) | set(validation["author"])

    history_posts = levelled[
        levelled["author"].isin(selected_authors)
    ].copy()

    history_posts = history_posts.sort_values(
        ["author", "posted_at"],
        kind="mergesort",
    ).reset_index(drop=True)

    recent_features = recent_history_features(history_posts)
    recent_columns = ["recent_last_resid", "recent_five_mean_resid"]

    history_tfidf = vectorizer.transform(history_posts["text"])
    history_vectors = svd.transform(history_tfidf)
    history_vectors = normalize(history_vectors)

    max_similarity = np.full(len(history_posts), np.nan)
    profile_similarity = np.full(len(history_posts), np.nan)
    similar_past_result = np.full(len(history_posts), np.nan)

    for author, group in history_posts.groupby("author", sort=False):
        positions = group.index.to_numpy()
        vectors = history_vectors[positions]

        has_text_vector = np.linalg.norm(vectors, axis=1) > 0

        for local_position, post in enumerate(group.itertuples()):
            if not has_text_vector[local_position]:
                continue

            earlier = (
                group["posted_at"].lt(post.posted_at).to_numpy()
                & has_text_vector
            )

            if not earlier.any():
                continue

            current_vector = vectors[local_position]
            past_vectors = vectors[earlier]
            position = positions[local_position]

            similarities = past_vectors @ current_vector
            max_similarity[position] = np.clip(
                similarities.max(), -1, 1
            )

            past_results = group["resid"].to_numpy()[earlier]
            valid_results = np.isfinite(past_results)

            if valid_results.any():
                eligible_similarities = similarities[valid_results]
                eligible_results = past_results[valid_results]

                top_indices = np.argsort(eligible_similarities)[-3:]

                similar_past_result[position] = (
                    eligible_results[top_indices].mean()
                )

            profile = past_vectors.mean(axis=0)
            profile_length = np.linalg.norm(profile)

            if profile_length > 0:
                profile_similarity[position] = np.clip(
                    (profile @ current_vector) / profile_length,
                    -1,
                    1,
                )

    history_posts["history_max_similarity"] = max_similarity
    history_posts["history_profile_similarity"] = profile_similarity
    history_posts["similar_past_result"] = similar_past_result

    history_columns = [
        "history_max_similarity",
        "history_profile_similarity",
        "similar_past_result",
    ]


    baseline_features = features.build(
        levelled,
        load_followers(data_dir),
    )

    baseline_features = baseline_features.merge(
        hook_features,
        on=['author', 'post'],
        how='left',
    )

    baseline_features = baseline_features.merge(
        repost_features,
        on=["author", "post"],
        how="left",
    )

    baseline_features = baseline_features.merge(
        profile_features,
        on="author",
        how="left",
    )

    baseline_features = baseline_features.merge(
        recent_features,
        on=["author", "post"],
        how="left",
    )

    baseline_features = baseline_features.merge(
        history_posts[["author", "post"] + history_columns],
        on=["author", "post"],
        how="left",
    )
    features_train = train[["author", "post"]].merge(
        baseline_features,
        on=["author", "post"],
        how="left",
    )

    features_val = validation[["author", "post"]].merge(
        baseline_features,
        on=["author", "post"],
        how="left",
    )


    text_columns = [
        f"text_svd_{number}" for number in range(100)
    ]

    features_train = pd.concat(
        [
            features_train,
            pd.DataFrame(
                compressed_train,
                columns=text_columns,
                index=features_train.index,
            ),
        ],
        axis=1,
    )

    features_val = pd.concat(
        [
            features_val,
            pd.DataFrame(
                compressed_val,
                columns=text_columns,
                index=features_val.index,
            ),
        ],
        axis=1,
    )

    numeric_columns = (
        features.CONTENT_NUM
        + features.HISTORY_NUM
        + features.ARTEFACT_NUM
        + text_columns
        + history_columns
        + profile_columns
        + recent_columns
        + repost_columns
        + hook_columns
    )

    model = make_baseline()
    counts = train["label"].value_counts()
    balanced_weights = len(train) / (3 * counts)

    moderate_weights = {}
    for label, weight in balanced_weights.items():
        moderate_weights[int(label)] = float(weight ** 0.5)

    model.set_params(clf__class_weight=moderate_weights)

    model.set_params(
        prep=ColumnTransformer([
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                ),
                features.CONTENT_CAT,
            ),
        ])
    )


    print('Train posts:', len(features_train))
    print('Validation posts:', len(features_val))
    print('Numeric features:', len(numeric_columns))
    print('Training...')
    model.fit(features_train, train["label"])

    predictions = model.predict(features_val)

    return score(validation["label"], predictions)
