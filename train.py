from pathlib import Path
from kairos.data import load_labels, load_posts

from sklearn.model_selection import train_test_split

from experiments.text_model import run as run_text
from experiments.baseline_model import run as run_baseline
from experiments.text_and_baseline import run as run_combined
from experiments.text_svd_baseline import run as run_svd
from experiments.text_svd_history import run as run_history
from experiments.text_svd_history_profile import run as run_profile
from experiments.text_svd_history_profile_text import run as run_profile_text
from experiments.text_svd_recent_history import run as run_recent
from experiments.text_embeddings import run as run_embeddings
from experiments.text_embeddings_profile import run as run_embedding_profile
from experiments.text_with_images import run as run_images
from experiments.text_with_video import run as run_video
from experiments.text_with_reposts import run as run_reposts
from experiments.text_with_hook import run as run_hook
from experiments.text_with_hook_embeddings import run as run_hook_e5
from experiments.text_with_author_style import run as run_author_style
from experiments.text_with_author_style_images import run as run_author_style_images
from experiments.text_with_reshared_text import run as run_reshared_text
from experiments.compare_classifiers import run as run_classifiers
from model import run as run_model

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

# history_metrics = run_history(train, validation, posts, DATA)
# print("svd history", history_metrics)

# profile_metrics = run_profile(train, validation, posts, DATA)
# print("svd history profi;e", profile_metrics)

# profile_text_metrics = run_profile_text(
#     train, validation, posts, DATA
# )
# print("svd + history +profile text", profile_text_metrics)

# recent_metrics = run_recent(train, validation, posts, DATA)
# print("with recent history", recent_metrics)

# embedding_metrics = run_embeddings(train, validation, posts, DATA)
# print(" embeddings", embedding_metrics)

# metrics = run_embedding_profile(train, validation, posts, DATA)
# print("E5 + profile", metrics)

# image_metrics = run_images(train, validation, posts, DATA)
# print("with images ", image_metrics)

# video_metrics = run_video(train, validation, posts, DATA)
# print("VIDEO", video_metrics)

# repost_metrics = run_reposts(train, validation, posts, DATA)
# print("repost ", repost_metrics)

# hook_metrics = run_hook(train, validation, posts, DATA)
# print("hooks", hook_metrics)
# metrics = run_hook_e5(train, validation, posts, DATA)
# print("hook e5", metrics)

# metrics = run_author_style(train, validation, posts, DATA)
# print("hook e5 + author style ", metrics)

# metrics = run_author_style_images(train, validation, posts, DATA)
# print("Hook E5 + author style + images:", metrics)

# metrics = run_reshared_text(train, validation, posts, DATA)
# print("+ reshared text", metrics)

# comparison = run_classifiers(train, validation, posts, DATA)
# print("compare classifiers", comparison)

# metrics = run_model(train, validation, posts, DATA)
# print("validation:", metrics)

metrics = run_model(train_pool, test, posts, DATA, use_test_hooks=True)
print("test:", metrics)
