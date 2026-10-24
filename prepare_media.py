from pathlib import Path
import csv
import io
import json
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kairos.data import load_labels


DATA = Path('/Volumes/caelmbleidd/lab1/dataset/core')
MANIFEST = DATA / "media" / "manifest.jsonl"
VIDEO_ROOT = Path('/Volumes/caelmbleidd/lab1/dataset/video/media')

OUTPUT = Path("/Users/ekaterina/kairos-dataset/results/media_clip_v1")
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
FRAME_FRACTIONS = [0.1, 0.5, 0.9]
LIMIT = 0


def select_media():
    labels = load_labels(DATA)
    authors = set(labels.loc[labels["split"] == "train", "author"])
    links = []
    seen = set()
    counts = {}

    with MANIFEST.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            kind = item.get("kind")
            if item.get("author") not in authors:
                continue
            if item.get("from_reshared_post") is not False:
                continue
            if kind not in {"post_image", "post_video"}:
                continue

            key = (item["author"], item["post"], kind, item["file"])
            if key in seen:
                continue
            seen.add(key)

            post_key = key[:3]
            maximum = 4 if kind == "post_image" else 1
            if counts.get(post_key, 0) >= maximum:
                continue
            counts[post_key] = counts.get(post_key, 0) + 1

            links.append({
                "author": item["author"],
                "post": item["post"],
                "kind": kind,
                "file": item["file"],
            })
    return links


def read_video_frames(path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid video duration")

    frames = []
    try:
        for fraction in FRAME_FRACTIONS:
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error",
                    "-ss", str(duration * fraction),
                    "-i", str(path), "-frames:v", "1",
                    "-vf", "scale=512:512:force_original_aspect_ratio=decrease",
                    "-f", "image2pipe", "-vcodec", "png", "pipe:1",
                ],
                capture_output=True,
                check=True,
                timeout=60,
            )
            with Image.open(io.BytesIO(result.stdout)) as frame:
                frames.append(frame.convert("RGB"))
    except BaseException:
        for frame in frames:
            frame.close()
        raise
    return frames


def process_file(kind, filename, model):
    root = DATA / "media" if kind == "post_image" else VIDEO_ROOT
    path = root / filename
    if kind == "post_image":
        with Image.open(path) as image:
            images = [ImageOps.exif_transpose(image).convert("RGB")]
    else:
        images = read_video_frames(path)

    try:
        vectors = model.encode(
            images,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    finally:
        for image in images:
            image.close()

    expected_rows = 1 if kind == "post_image" else len(FRAME_FRACTIONS)
    if vectors.shape != (expected_rows, 512) or not np.isfinite(vectors).all():
        raise ValueError("error CLIP vectors")
    return vectors.astype(np.float32)


def main():
    links = select_media()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "links.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=["author", "post", "kind", "file"]
        )
        writer.writeheader()
        writer.writerows(links)

    jobs = list(dict.fromkeys((item["kind"], item["file"]) for item in links))
    print("Media files:", len(jobs))

    pending = []
    for kind, filename in jobs:
        target = OUTPUT / kind / f"{Path(filename).stem}.npz"
        if not target.is_file():
            pending.append((kind, filename, target))

    print("remaining:", len(pending))
    if LIMIT:
        images = [job for job in pending if job[0] == "post_image"]
        videos = [job for job in pending if job[0] == "post_video"]
        pending = images[:LIMIT] + videos[:LIMIT]
    if not pending:
        print("all done")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    started = time.perf_counter()
    successful = 0
    failed = 0

    for kind, filename, target in tqdm(pending, desc="media", unit="file"):
        try:
            vectors = process_file(kind, filename, model)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failed += 1
            with (OUTPUT / "errors.jsonl").open("a", encoding="utf-8") as destination:
                destination.write(json.dumps({
                    "kind": kind, "file": filename, "error": str(error),
                }, ensure_ascii=False) + "\n")
            tqdm.write(f"error: {filename}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as destination:
            np.savez_compressed(
                destination,
                vectors=vectors,
                model_name=np.array(MODEL_NAME),
                kind=np.array(kind),
                source_file=np.array(filename),
                frame_fractions=np.array(
                    FRAME_FRACTIONS if kind == "post_video" else []
                ),
            )
        temporary.replace(target)
        successful += 1

    print(f"successes {successful}, errors {failed}")
    print(f"time {(time.perf_counter() - started) / 60:.1f} min.")
    print("saved", OUTPUT)


if __name__ == "__main__":
    main()
