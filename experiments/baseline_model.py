from baseline import model as make_baseline
from kairos import features
from kairos.data import load_followers
from kairos.label import author_level
from kairos.metrics import score


def run(train, validation, posts, data_dir):
    levelled = author_level(posts)
    baseline_features = features.build(
        levelled,
        load_followers(data_dir),
    )

    baseline_train = train[["author", "post", "label"]].merge(
        baseline_features,
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )

    baseline_val = validation[["author", "post", "label"]].merge(
        baseline_features,
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )

    assert baseline_train["level"].notna().all()
    assert baseline_val["level"].notna().all()

    model = make_baseline()
    model.fit(baseline_train, baseline_train["label"])
    predictions = model.predict(baseline_val)

    return score(baseline_val["label"], predictions)