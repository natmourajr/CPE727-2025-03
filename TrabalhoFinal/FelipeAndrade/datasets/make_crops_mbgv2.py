import os
import csv
from collections import Counter
import xml.etree.ElementTree as ET

import cv2
import numpy as np

MBGV2_ROOT = "/home/felipe.andrade/data/mbgv2/v2"
ANNOTATIONS_DIR = os.path.join(MBGV2_ROOT, "annotations-xml")
FRAMES_DIR = os.path.join(MBGV2_ROOT, "frames")

VIDEO_IDS = [
    10, 11, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35,
    36, 37,
]
VIDEO_NAMES = {f"video{vid}" for vid in VIDEO_IDS}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
OUT_ROOT = os.path.join(BASE_DIR, "data", "crops")
OUT_IMAGES_DIR = os.path.join(OUT_ROOT, "images")
METADATA_PATH = os.path.join(OUT_ROOT, "metadata.csv")

CROP_SIZE = 128
MAX_PER_CLASS = None


def parse_cvat_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    objects = []

    for track in root.findall("track"):
        label = track.get("label")
        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            objects.append(
                {
                    "label": label,
                    "frame": frame,
                    "bbox": (xtl, ytl, xbr, ybr),
                }
            )
    return objects


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def main():
    print("MBGv2: gerando crops de bounding boxes...")
    print("Base:", MBGV2_ROOT)
    print("Saída de imagens:", OUT_IMAGES_DIR)
    print("Metadata CSV:", METADATA_PATH)

    if not os.path.isdir(ANNOTATIONS_DIR):
        raise RuntimeError(f"Diretório de anotações não encontrado: {ANNOTATIONS_DIR}")

    ensure_dir(OUT_IMAGES_DIR)
    ensure_dir(os.path.dirname(METADATA_PATH))

    csv_file = open(METADATA_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["filepath", "label", "video", "frame", "xtl", "ytl", "xbr", "ybr"]
    )

    class_counter = Counter()
    used_per_class = Counter()

    xml_files = sorted(f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml"))

    total_skipped = 0
    total_written = 0

    for fname in xml_files:
        stem = os.path.splitext(fname)[0]  # ex: video10
        if stem not in VIDEO_NAMES:
            continue

        xml_path = os.path.join(ANNOTATIONS_DIR, fname)
        print(f"Processando anotações de {stem} ({xml_path})...")
        objects = parse_cvat_xml(xml_path)

        for idx, obj in enumerate(objects):
            label = obj["label"]
            frame_idx = obj["frame"]

            if MAX_PER_CLASS is not None and used_per_class[label] >= MAX_PER_CLASS:
                continue

            # padrão frame_0000.png
            frame_filename = f"frame_{frame_idx:04d}.png"
            frame_path = os.path.join(FRAMES_DIR, stem, frame_filename)

            if not os.path.exists(frame_path):
                total_skipped += 1
                continue

            img = cv2.imread(frame_path)
            if img is None:
                print(f"Aviso: não foi possível ler {frame_path}")
                total_skipped += 1
                continue

            h, w = img.shape[:2]
            xtl, ytl, xbr, ybr = obj["bbox"]
            x1, y1, x2, y2 = clamp_bbox(xtl, ytl, xbr, ybr, w, h)

            if x2 <= x1 or y2 <= y1:
                total_skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                total_skipped += 1
                continue

            crop_resized = cv2.resize(
                crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
            )

            class_dir = os.path.join(OUT_IMAGES_DIR, label)
            ensure_dir(class_dir)

            out_name = f"{stem}_f{frame_idx:06d}_o{idx:05d}.jpg"
            out_path = os.path.join(class_dir, out_name)

            success = cv2.imwrite(out_path, crop_resized)
            if not success:
                print(f"Aviso: falha ao salvar {out_path}")
                total_skipped += 1
                continue

            rel_path = os.path.relpath(out_path, BASE_DIR)

            writer.writerow(
                [
                    rel_path,
                    label,
                    stem,
                    frame_idx,
                    xtl,
                    ytl,
                    xbr,
                    ybr,
                ]
            )

            class_counter[label] += 1
            used_per_class[label] += 1
            total_written += 1

    csv_file.close()

    print("\nResumo:")
    print("Total de crops salvos:", total_written)
    print("Total de boxes ignoradas:", total_skipped)
    print("Crops por classe:")
    for cls, cnt in class_counter.most_common():
        print(f"  {cls}: {cnt}")

    if MAX_PER_CLASS is not None:
        print("\nObservação: limite por classe (MAX_PER_CLASS) =", MAX_PER_CLASS)


if __name__ == "__main__":
    main()
