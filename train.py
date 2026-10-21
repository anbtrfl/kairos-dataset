from pathlib import Path
from kairos.data import load_labels, load_posts

from sklearn.model_selection import train_test_split

from experiments.text_model import run as run_text
from experiments.baseline_model import run as run_baseline
from experiments.text_and_baseline import run as run_combined
from experiments.text_svd_baseline import run as run_svd
from experiments.text_svd_history import run as run_history

DATA = Path('/Volumes/caelmbleidd/lab1/dataset/core')

labels = load_labels(DATA)
posts = load_posts(DATA)

table = labels.merge(
    posts,
    on=["author", "post"],
    how="left",
    validate="one_to_one",
)

# print(f"posts all - {len(table)}")
# print("split train/test -")
# print(table["split"].value_counts())

# print("no text ")
# print(table["text"].isna().sum())

train_pool = table[table["split"] == "train"].copy()
test = table[table["split"] == "test"].copy()

authors = sorted(train_pool["author"].unique())
train_authors, val_authors = train_test_split(
    authors,
    test_size=0.2,
    random_state=42,
)

train = train_pool[
    train_pool["author"].isin(train_authors)
].copy()

validation = train_pool[
    train_pool["author"].isin(val_authors)
].copy()

# print("sets - ")
# for name, part in [
#     ("train", train),
#     ("validation", validation),
#     ("test", test),
# ]:
#     print(
#         f"{name}: {len(part)} posts, "
#         f"{part['author'].nunique()} authors"
#     )

# train_ids = set(train["author"])
# val_ids = set(validation["author"])
# test_ids = set(test["author"])

# assert train_ids.isdisjoint(val_ids)
# assert train_ids.isdisjoint(test_ids)
# assert val_ids.isdisjoint(test_ids)

# print("no peresecheniy")

# text_metrics = run_text(train, validation)
# print("tf-idf only - ", text_metrics)

# baseline_metrics = run_baseline(train, validation, posts, DATA)
# print("baseline - ", baseline_metrics)

# combined_metrics = run_combined(train, validation, posts, DATA)
# print("combined", combined_metrics)

# svd_metrics = run_svd(train, validation, posts, DATA)
# print("SVD + baseline:", svd_metrics)

history_metrics = run_history(train, validation, posts, DATA)
print("SVD + история:", history_metrics)