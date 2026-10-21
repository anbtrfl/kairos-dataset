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


def run(train, validation, posts, data_dir):
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

    levelled = author_level(posts)

    selected_authors = set(train["author"]) | set(validation["author"])

    history_posts = levelled[
        levelled["author"].isin(selected_authors)
    ].copy()

    history_posts = history_posts.sort_values(
        ["author", "posted_at"],
        kind="mergesort",
    ).reset_index(drop=True)

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

    print(history_posts[history_columns].describe())

    baseline_features = features.build(
        levelled,
        load_followers(data_dir),
    )

    baseline_features = baseline_features.merge(
        history_posts[["author", "post"] + history_columns],
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )
    features_train = train[["author", "post"]].merge(
        baseline_features,
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )

    features_val = validation[["author", "post"]].merge(
        baseline_features,
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )

    assert features_train["level"].notna().all()
    assert features_val["level"].notna().all()

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
    )

    model = make_baseline()

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

    model.fit(features_train, train["label"])

    predictions = model.predict(features_val)

    return score(validation["label"], predictions)