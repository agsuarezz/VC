"""Prepare JAFFE subject-disjoint data or original FER2013 CSV, without redistribution."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from lebron.emotion import LABELS, ExpressionFaceCropper

JAFFE_URL = "https://zenodo.org/records/14974867/files/jaffe.zip?download=1"
JAFFE_MD5 = "fe13f3302eb9968ef04367456f665436"
CODES = {"HA": "happy", "SA": "sad", "AN": "angry", "SU": "surprise", "NE": "neutral"}
FER_CODES = {0: "angry", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}


def jaffe_rows(source, seed):
    """Crop with the same OpenCV face detector family used by the application."""
    detector = ExpressionFaceCropper()
    files = sorted(Path(source).rglob("*.tiff")) + sorted(Path(source).rglob("*.tif"))
    selected = [(p, p.name.split(".")) for p in files]
    selected = [(p, parts) for p, parts in selected if len(parts) > 2 and parts[1][:2] in CODES]
    subjects = sorted({parts[0] for _, parts in selected})
    if len(subjects) != 10:
        raise ValueError(f"Se esperaban 10 sujetos JAFFE; encontrados {len(subjects)}")
    np.random.default_rng(seed).shuffle(subjects)
    split_map = {s: "train" if i < 6 else "val" if i < 8 else "test" for i, s in enumerate(subjects)}
    rows, rejected = [], []
    for path, parts in selected:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected.append(path.name)
            continue
        box = detector.detect(image)
        if box is None:
            rejected.append(path.name)
            continue
        x, y, w, h = box
        face = cv2.resize(image[y:y+h, x:x+w], (64, 64), interpolation=cv2.INTER_AREA)
        rows.append((face, CODES[parts[1][:2]], split_map[parts[0]], parts[0], path.name))
    return rows, {"subject_splits": split_map, "rejected_no_face_or_unreadable": rejected,
                  "source": "JAFFE official v2", "split_strategy": "6/2/2 disjoint subjects"}


def fer_rows(source):
    rows = []
    usage_map = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    with Path(source).open(newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            code = int(row["emotion"])
            if code not in FER_CODES:
                continue
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
            if pixels.size != 2304 or row["Usage"] not in usage_map:
                raise ValueError(f"Fila FER2013 inválida: {i}")
            face = cv2.resize(pixels.reshape(48, 48), (64, 64), interpolation=cv2.INTER_AREA)
            rows.append((face, FER_CODES[code], usage_map[row["Usage"]], "unknown", f"row-{i}"))
    return rows, {"source": "FER2013 original CSV supplied by user", "split_strategy": "official Usage",
                  "limitation": "No subject IDs; subject overlap cannot be excluded"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["jaffe", "fer2013"], default="jaffe")
    p.add_argument("--source", type=Path, required=True, help="JAFFE directory or original FER2013 CSV")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--download-jaffe", action="store_true", help="Download from official source into --source")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.download_jaffe:
        if args.dataset != "jaffe":
            p.error("--download-jaffe requires --dataset jaffe")
        args.source.mkdir(parents=True, exist_ok=True)
        archive = args.source / "jaffe.zip"
        urllib.request.urlretrieve(JAFFE_URL, archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != JAFFE_MD5:
            raise ValueError("El checksum del archivo JAFFE no coincide con el oficial")
        with zipfile.ZipFile(archive) as z:
            base = args.source.resolve()
            for name in z.namelist():
                if not (base / name).resolve().is_relative_to(base):
                    raise ValueError("Ruta insegura en ZIP")
            z.extractall(base)
    rows, metadata = jaffe_rows(args.source, args.seed) if args.dataset == "jaffe" else fer_rows(args.source)
    # Test has priority: remove exact pixel duplicates from train/val to avoid leakage.
    priority = {"test": 0, "val": 1, "train": 2}
    seen, clean, removed = set(), [], []
    for row in sorted(rows, key=lambda r: priority[r[2]]):
        digest = hashlib.sha256(row[0].tobytes()).hexdigest()
        if digest in seen:
            removed.append(row[4])
            continue
        seen.add(digest)
        clean.append(row)
    if not clean:
        raise ValueError("No hay imágenes válidas")
    for split in ("train", "val", "test"):
        if set(r[1] for r in clean if r[2] == split) != set(LABELS):
            raise ValueError(f"Faltan clases en {split}; revisa los datos")
    metadata.update(seed=args.seed, removed_exact_duplicates=removed,
                    counts={s: dict(Counter(r[1] for r in clean if r[2] == s)) for s in ("train", "val", "test")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=np.stack([r[0] for r in clean]),
                        labels=np.array([r[1] for r in clean]), splits=np.array([r[2] for r in clean]),
                        subjects=np.array([r[3] for r in clean]), names=np.array([r[4] for r in clean]),
                        metadata=np.array(json.dumps(metadata)))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
