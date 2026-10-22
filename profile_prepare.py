from pathlib import Path
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from kairos.data import load_labels
from experiments.text_svd_history_profile_text import load_profile_texts

DATA = Path('/Volumes/caelmbleidd/lab1/dataset/core')

labels = load_labels(DATA)

train_authors = labels.loc[
    labels["split"] == "train", "author"
].unique()

profiles = load_profile_texts(DATA)

selected = profiles[
    profiles["author"].isin(train_authors)
].copy().reset_index(drop=True)

assert selected["author"].is_unique
assert set(selected["author"]) == set(train_authors)
assert selected["profile_text"].notna().all()

has_text = selected["profile_text"].str.strip().ne("").to_numpy()

# print("all", len(selected))
# print("with text", has_text.sum())

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device=device,
)

vectors = np.zeros(
    (len(selected), model.get_embedding_dimension()),
    dtype=np.float32,
)

texts = [
    "query: " + text
    for text in selected.loc[has_text, "profile_text"]
]

started = time.perf_counter()

if texts:
    vectors[has_text] = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

assert np.isfinite(vectors).all()

output = Path("results/embeddings/profiles_e5_small.npz")
output.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    output,
    author=selected["author"].to_numpy(dtype=str),
    vectors=vectors,
    has_text=has_text,
    model_name=np.array("intfloat/multilingual-e5-small"),
    prefix=np.array("query: "),
    max_seq_length=np.array(model.max_seq_length),
)

elapsed = time.perf_counter() - started

# print("size vectors", vectors.shape)
# print(f"time {elapsed / 60:.1f} min.")
