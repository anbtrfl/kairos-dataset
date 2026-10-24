from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

CACHE = Path("/Users/ekaterina/kairos-dataset/results/media_clip_v1")

def aggregate(kind, output_name):
    links = pd.read_csv(CACHE / "links.csv")
    links = links[links["kind"] == kind].drop_duplicates(
        ["author", "post", "file"]
    )

    authors = []
    posts = []
    vectors = []
    expected_counts = []
    loaded_counts = []
    missing_files = set()

    groups = links.groupby(["author", "post"], sort=False)

    for (author, post), group in tqdm(
        groups,
        total=groups.ngroups,
        desc=kind,
    ):
        file_vectors = []

        for filename in group["file"]:
            path = CACHE / kind / f"{Path(filename).stem}.npz"

            if not path.is_file():
                missing_files.add(filename)
                continue

            with np.load(path, allow_pickle=False) as saved:

                values = saved["vectors"]


            file_vectors.append(values.mean(axis=0))

        vector = np.zeros(512, dtype=np.float32)

        if file_vectors:
            vector = np.mean(file_vectors, axis=0)
            length = np.linalg.norm(vector)

            if length > 0:
                vector = vector / length

        authors.append(author)
        posts.append(post)
        vectors.append(vector)
        expected_counts.append(len(group))
        loaded_counts.append(len(file_vectors))

    matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, 512)
    output = CACHE / output_name

    np.savez_compressed(
        output,
        author=np.asarray(authors, dtype=str),
        post=np.asarray(posts, dtype=str),
        vectors=matrix,
        expected_count=np.asarray(expected_counts),
        loaded_count=np.asarray(loaded_counts),
        model_name=np.array("sentence-transformers/clip-ViT-B-32"),
        kind=np.array(kind),
    )

    print("size -", matrix.shape)
    print("posts without", sum(
        count == 0 for count in loaded_counts
    ))
    print("missing files", len(missing_files))
    print("saved - ", output.name)


if __name__ == "__main__":
    aggregate("post_image", "images_by_post.npz")
    aggregate("post_video", "videos_by_post.npz")
