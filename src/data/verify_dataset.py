"""
src/data/verify_dataset.py
──────────────────────────
Utilitas untuk memeriksa integritas, kelengkapan, dan distribusi dataset YOLOv11.

Fungsi pemeriksaan:
1. Memverifikasi file konfigurasi data.yaml dan resolusi direktori split (train, val, test).
2. Memeriksa ketersediaan gambar dan korespondensi file label 1:1.
3. Mendeteksi file label kosong (gambar latar belakang/background).
4. Menghitung total anotasi dan distribusi sebaran objek per kelas (A-Z).
5. Memvalidasi rentang koordinat bounding box YOLO (0.0 - 1.0).

Penggunaan:
-----------
    # Menjalankan pemeriksaan ringkas:
    python -m src.data.verify_dataset

    # Menjalankan dengan rincian per kelas:
    python -m src.data.verify_dataset --detail

    # Menentukan path custom data.yaml:
    python -m src.data.verify_dataset --data datasets/bisindo-dataset-1/data.yaml
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


def load_yaml(yaml_path: Path) -> dict:
    """Membaca file YAML konfigurasi dataset."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_path(base_dir: Path, rel_path: str) -> Path:
    """Menyelesaikan path relatif split dari lokasi data.yaml."""
    p1 = (base_dir / rel_path).resolve()
    if p1.exists():
        return p1
    # Jika format Roboflow menggunakan '../' tetapi foldernya ada di dalam base_dir
    clean_rel = rel_path.replace("../", "").lstrip("/\\")
    p2 = (base_dir / clean_rel).resolve()
    if p2.exists():
        return p2
    return p1


def verify_dataset(
    data_yaml_path: str = "datasets/bisindo-dataset-1/data.yaml",
    show_detail: bool = False,
) -> bool:
    """
    Menjalankan verifikasi kelengkapan dan integritas dataset YOLO.

    Returns:
        bool: True jika seluruh pemeriksaan lolos tanpa kesalahan kritis.
    """
    yaml_file = Path(data_yaml_path)
    print("=" * 68)
    print("           VERIFIKASI INTEGRITAS DATASET YOLOv11")
    print("=" * 68)
    print(f"File konfigurasi : {yaml_file}")

    if not yaml_file.exists():
        print(f"[ERROR] File konfigurasi '{yaml_file}' tidak ditemukan!")
        return False

    config = load_yaml(yaml_file)
    classes = config.get("names", [])
    num_classes = config.get("nc", len(classes))
    base_dir = yaml_file.parent

    print(f"Total kelas      : {num_classes}")
    print(f"Daftar kelas     : {', '.join(str(c) for c in classes[:10])}... (total {len(classes)} kelas)")
    print("-" * 68)

    splits = {
        "train": config.get("train", "train/images"),
        "val": config.get("val", "valid/images"),
        "test": config.get("test", "test/images"),
    }

    all_ok = True
    split_stats = {}
    total_class_counts = Counter()
    split_class_counts = defaultdict(Counter)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_name, split_rel in splits.items():
        if not split_rel:
            continue

        img_dir = resolve_split_path(base_dir, split_rel)
        if not img_dir.exists():
            print(f"[GAGAL] Direktori gambar untuk '{split_name}' tidak ditemukan: {img_dir}")
            all_ok = False
            continue

        # Direktori label biasanya berdampingan dengan images/
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            print(f"[GAGAL] Direktori label untuk '{split_name}' tidak ditemukan: {lbl_dir}")
            all_ok = False
            continue

        # Pindai file gambar
        image_files = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_extensions]
        label_files = {p.stem: p for p in lbl_dir.glob("*.txt")}

        missing_labels = []
        empty_labels = 0
        corrupt_boxes = 0
        instance_count = 0

        for img_path in image_files:
            stem = img_path.stem
            if stem not in label_files:
                missing_labels.append(img_path.name)
                continue

            lbl_path = label_files[stem]
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                empty_labels += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    corrupt_boxes += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])

                    # Validasi batas normalized bounding box [0, 1]
                    if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        corrupt_boxes += 1

                    total_class_counts[cls_id] += 1
                    split_class_counts[split_name][cls_id] += 1
                    instance_count += 1
                except ValueError:
                    corrupt_boxes += 1

        split_stats[split_name] = {
            "images": len(image_files),
            "labels": len(label_files),
            "instances": instance_count,
            "empty": empty_labels,
            "missing": len(missing_labels),
            "corrupt": corrupt_boxes,
        }

        if missing_labels:
            all_ok = False

    # Tampilkan Ringkasan Tabel
    print(f"{'Split':<10} {'Gambar':<10} {'Label':<10} {'Objek/Box':<12} {'Latar Kosong':<14} {'Status':<10}")
    print("-" * 68)
    total_imgs = 0
    total_lbls = 0
    total_instances = 0

    for s_name, st in split_stats.items():
        total_imgs += st["images"]
        total_lbls += st["labels"]
        total_instances += st["instances"]
        status = "Lolos" if st["missing"] == 0 and st["corrupt"] == 0 else "Perhatian"
        print(f"{s_name.upper():<10} {st['images']:<10} {st['labels']:<10} {st['instances']:<12} {st['empty']:<14} {status:<10}")

    print("-" * 68)
    print(f"{'TOTAL':<10} {total_imgs:<10} {total_lbls:<10} {total_instances:<12}")
    print("=" * 68)

    # Tampilkan rincian per kelas jika diminta
    if show_detail:
        print("\n=== DISTRIBUSI ANOTASI PER KELAS ALFABET BISINDO ===")
        print(f"{'ID':<4} {'Nama Kelas':<12} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("-" * 58)
        for cid in range(num_classes):
            cname = classes[cid] if cid < len(classes) else f"Class_{cid}"
            tr = split_class_counts["train"][cid]
            vl = split_class_counts["val"][cid]
            ts = split_class_counts["test"][cid]
            tot = total_class_counts[cid]
            print(f"{cid:<4} {str(cname):<12} {tr:<10} {vl:<10} {ts:<10} {tot:<10}")
        print("=" * 58)

    if all_ok:
        print("\n[SUKSES] Dataset utuh dan siap digunakan untuk pelatihan YOLOv11!")
    else:
        print("\n[PERINGATAN] Ditemukan ketidaksesuaian label atau file yang hilang. Periksa rincian di atas.")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifikasi integritas dataset YOLOv11 BISINDO.")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/bisindo-dataset-1/data.yaml",
        help="Path ke file data.yaml (default: datasets/bisindo-dataset-1/data.yaml)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Menampilkan rincian sebaran per kelas (A-Z).",
    )
    args = parser.parse_args()

    success = verify_dataset(data_yaml_path=args.data, show_detail=args.detail)
    sys.exit(0 if success else 1)
