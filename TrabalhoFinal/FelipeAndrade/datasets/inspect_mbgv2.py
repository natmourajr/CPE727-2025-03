import os
import xml.etree.ElementTree as ET
from collections import Counter

# Caminho base do MBGv2
MBGV2_ROOT = "/home/felipe.andrade/data/mbgv2/v2"
ANNOTATIONS_DIR = os.path.join(MBGV2_ROOT, "annotations-xml")

# Vídeos
VIDEO_IDS = [
    10, 11, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35,
    36, 37,
]
VIDEO_NAMES = {f"video{vid}" for vid in VIDEO_IDS}


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


def main():
    if not os.path.isdir(ANNOTATIONS_DIR):
        raise RuntimeError(f"Diretório de anotações não encontrado: {ANNOTATIONS_DIR}")

    class_counter = Counter()
    total_boxes = 0

    xml_files = sorted(
        f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")
    )

    if not xml_files:
        raise RuntimeError(f"Nenhum XML encontrado em {ANNOTATIONS_DIR}")

    for fname in xml_files:
        # Ex: video10.xml -> video10
        stem = os.path.splitext(fname)[0]
        if stem not in VIDEO_NAMES:
            # Ignora qualquer XML fora da lista
            continue

        path = os.path.join(ANNOTATIONS_DIR, fname)
        objs = parse_cvat_xml(path)
        total_boxes += len(objs)
        class_counter.update(o["label"] for o in objs)

    print("=== MBGv2: inspeção das anotações ===")
    print("Diretório de anotações:", ANNOTATIONS_DIR)
    print("Vídeos considerados:", sorted(VIDEO_NAMES))
    print()
    print("Total de bounding boxes:", total_boxes)
    print("Classes e contagens:")
    for cls, cnt in class_counter.most_common():
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()