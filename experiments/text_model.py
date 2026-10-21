from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from kairos.metrics import score


def run(train, validation):
    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(validation["text"])

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, train["label"])
    predictions = model.predict(X_val)

    return score(validation["label"], predictions)