from sklearn.feature_extraction.text import TfidfVectorizer

from kairos import features
from kairos.data import load_followers
from kairos.label import author_level

from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from kairos.metrics import score

def run(train, validation, posts, data_dir):
    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    text_train = vectorizer.fit_transform(train["text"])
    text_val = vectorizer.transform(validation["text"])

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

    # print("text train:", text_train.shape)
    # print("features baseline train:", features_train.shape)

    numeric_columns = (
        features.CONTENT_NUM
        + features.HISTORY_NUM
        + features.ARTEFACT_NUM
    )

    numeric_preparation = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    preparation = ColumnTransformer([
        ("numbers", numeric_preparation, numeric_columns),
        (
            "post_type",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
            ),
            features.CONTENT_CAT,
        ),
    ])

    extra_train = preparation.fit_transform(features_train)
    extra_val = preparation.transform(features_val)

    X_train = hstack([text_train, extra_train], format="csr")
    X_val = hstack([text_val, extra_val], format="csr")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, train["label"])
    predictions = model.predict(X_val)

    return score(validation["label"], predictions)