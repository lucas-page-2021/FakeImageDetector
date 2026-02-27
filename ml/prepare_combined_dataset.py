#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


HEADER = ["", "original_path", "id", "label", "label_str", "path"]


def read_rvf_rows(csv_path: Path, rvf_root_prefix: str):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = (row.get("label_str") or "").strip()
            rel_path = (row.get("path") or "").strip()
            if not label_str or not rel_path:
                continue

            label = 1 if label_str == "real" else 0
            rows.append(
                {
                    "original_path": row.get("original_path", ""),
                    "id": row.get("id", ""),
                    "label": label,
                    "label_str": "real" if label == 1 else "fake",
                    "path": f"{rvf_root_prefix}/{rel_path}",
                }
            )
    return rows


def build_zenodo_rows(metadata_path: Path, image_root: Path, image_prefix: str):
    data = json.loads(metadata_path.read_text())
    rows = []
    for key, item in data.items():
        image_path = image_root / f"{key}.png"
        if not image_path.exists():
            continue

        is_real = item.get("model") is None
        label = 1 if is_real else 0
        label_str = "real" if is_real else "fake"
        model_name = "ffhq" if is_real else str(item.get("model"))

        rows.append(
            {
                "original_path": f"zenodo://15121401/images/{key}.png",
                "id": f"zenodo15121401_{key}",
                "label": label,
                "label_str": label_str,
                "path": f"{image_prefix}/{key}.png",
                "model_name": model_name,
            }
        )
    return rows


def stratified_split(rows, valid_ratio: float, seed: int):
    real = [r for r in rows if r["label"] == 1]
    fake = [r for r in rows if r["label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(fake)

    real_valid_n = int(round(len(real) * valid_ratio))
    fake_valid_n = int(round(len(fake) * valid_ratio))

    valid = real[:real_valid_n] + fake[:fake_valid_n]
    train = real[real_valid_n:] + fake[fake_valid_n:]
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for idx, row in enumerate(rows):
            writer.writerow([idx, row["original_path"], row["id"], row["label"], row["label_str"], row["path"]])


def count_by_label(rows):
    real = sum(1 for r in rows if r["label"] == 1)
    fake = sum(1 for r in rows if r["label"] == 0)
    return {"real": real, "fake": fake, "total": real + fake}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvf-train-csv", default="datasets/Real vs Fake Faces - 10k/train.csv")
    parser.add_argument("--rvf-valid-csv", default="datasets/Real vs Fake Faces - 10k/valid.csv")
    parser.add_argument("--rvf-prefix", default="datasets/Real vs Fake Faces - 10k/rvf10k")
    parser.add_argument("--zenodo-metadata", default="datasets/zenodo-15121401/meta/metadata.json")
    parser.add_argument("--zenodo-images", default="datasets/zenodo-15121401/raw/images")
    parser.add_argument("--zenodo-prefix", default="datasets/zenodo-15121401/raw/images")
    parser.add_argument("--zenodo-valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-train-csv", default="datasets/combined_faces/train.csv")
    parser.add_argument("--out-valid-csv", default="datasets/combined_faces/valid.csv")
    parser.add_argument("--out-summary", default="datasets/combined_faces/summary.json")
    args = parser.parse_args()

    rvf_train_rows = read_rvf_rows(Path(args.rvf_train_csv), args.rvf_prefix)
    rvf_valid_rows = read_rvf_rows(Path(args.rvf_valid_csv), args.rvf_prefix)

    zenodo_rows = build_zenodo_rows(Path(args.zenodo_metadata), Path(args.zenodo_images), args.zenodo_prefix)
    zen_train_rows, zen_valid_rows = stratified_split(zenodo_rows, args.zenodo_valid_ratio, args.seed)

    all_train = rvf_train_rows + zen_train_rows
    all_valid = rvf_valid_rows + zen_valid_rows

    rng = random.Random(args.seed)
    rng.shuffle(all_train)
    rng.shuffle(all_valid)

    write_rows(Path(args.out_train_csv), all_train)
    write_rows(Path(args.out_valid_csv), all_valid)

    summary = {
        "rvf_train": count_by_label(rvf_train_rows),
        "rvf_valid": count_by_label(rvf_valid_rows),
        "zenodo_total": count_by_label(zenodo_rows),
        "zenodo_train": count_by_label(zen_train_rows),
        "zenodo_valid": count_by_label(zen_valid_rows),
        "combined_train": count_by_label(all_train),
        "combined_valid": count_by_label(all_valid),
        "seed": args.seed,
        "zenodo_valid_ratio": args.zenodo_valid_ratio,
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
