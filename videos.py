import json
from pathlib import Path
import subprocess
import time
import csv

MANIFEST = Path('/Volumes/caelmbleidd/lab1/dataset/core/media/manifest.jsonl')

videos = []
seen = set()

with MANIFEST.open(encoding="utf-8") as source:
    for line in source:
        if not line.strip():
            continue

        item = json.loads(line)

        if item.get("kind") != "post_video":
            continue

        if item.get("from_reshared_post") is not False:
            continue

        key = (item["author"], item["post"], item["file"])

        if key in seen:
            continue

        seen.add(key)
        videos.append(item)

print(f"all: {len(videos)}")
print(f"unique videos: {len({v['file'] for v in videos})}")

VIDEO_ROOT = Path('/Volumes/caelmbleidd/lab1/dataset/video/media')

unique_files = list(dict.fromkeys(item["file"] for item in videos))

output = Path("results/video_metadata.csv")
output.parent.mkdir(parents=True, exist_ok=True)

started = time.perf_counter()
successful = 0
failed = 0

with output.open("w", newline="", encoding="utf-8") as destination:
    writer = csv.DictWriter(
        destination,
        fieldnames=["file", "duration", "width", "height", "error"],
    )
    writer.writeheader()

    for number, filename in enumerate(unique_files, start=1):
        path = VIDEO_ROOT / filename
        row = {
            "file": filename,
            "duration": "",
            "width": "",
            "height": "",
            "error": "",
        }

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height:format=duration",
                    "-of", "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            info = json.loads(result.stdout)
            stream = info["streams"][0]

            row["duration"] = float(info["format"]["duration"])
            row["width"] = stream["width"]
            row["height"] = stream["height"]
            successful += 1

        except (
            subprocess.SubprocessError,
            OSError,
            ValueError,
            KeyError,
            IndexError,
        ) as error:
            row["error"] = str(error)
            failed += 1
            print(f"error{filename}")

        writer.writerow(row)

        if number % 100 == 0:
            destination.flush()
            print(f"done {number}/{len(unique_files)}")

elapsed = time.perf_counter() - started

print(f"successes - {successful}, errors - {failed}")
print(f"time - {elapsed / 60:.1f} min.")
print(f"Тtable saved in {output}")