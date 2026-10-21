from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import model as make_baseline
from kairos import features
from kairos.data import load_followers
from kairos.label import author_level
from kairos.metrics import score
import pandas as pd


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

    baseline_features = features.build(
        author_level(posts),
        load_followers(data_dir),
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