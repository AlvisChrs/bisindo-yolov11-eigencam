"""
src/data/plot_distribution.py
─────────────────────────────
Utilitas untuk menghasilkan grafik visualisasi sebaran data per kelas (A-Z)
pada dataset BISINDO untuk seluruh split (Train, Validation, Test).

Grafik yang dihasilkan berstandar publikasi ilmiah (300 DPI) dan dapat langsung
disematkan pada naskah skripsi (Bab 3 Metodologi Penelitian atau Bab 4 Hasil).

Penggunaan:
-----------
    # Menghasilkan grafik distribusi default:
    python -m src.data.plot_distribution

    # Menyimpan ke path custom:
    python -m src.data.plot_distribution --output results/figures/sebaran_bisindo.png
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml


def resolve_split_path(base_dir: Path, rel_path: str) -> Path:
    """Menyelesaikan path relatif split dari lokasi data.yaml."""
    p1 = (base_dir / rel_path).resolve()
    if p1.exists():
        return p1
    clean_rel = rel_path.replace("../", "").lstrip("/\\")
    p2 = (base_dir / clean_rel).resolve()
    if p2.exists():
        return p2
    return p1


def plot_class_distribution(
    data_yaml_path: str = "datasets/bisindo-dataset-1/data.yaml",
    output_path: str = "results/figures/class_distribution.png",
) -> None:
    """Membaca anotasi dataset dan menghasilkan grafik batang sebaran kelas."""
    yaml_file = Path(data_yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classes = [str(c).upper() for c in config.get("names", [])]
    num_classes = len(classes)
    base_dir = yaml_file.parent

    splits = {
        "Train": config.get("train", "train/images"),
        "Validation": config.get("val", "valid/images"),
        "Test": config.get("test", "test/images"),
    }

    counts = {s: Counter() for s in splits}

    print("[INFO] Memindai file label anotasi...")
    for split_name, split_rel in splits.items():
        if not split_rel:
            continue
        img_dir = resolve_split_path(base_dir, split_rel)
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls_id = int(parts[0])
                            counts[split_name][cls_id] += 1
                        except ValueError:
                            pass

    # Persiapan Data untuk Matplotlib
    y_indices = np.arange(num_classes)
    bar_height = 0.26

    train_vals = [counts["Train"][i] for i in range(num_classes)]
    val_vals = [counts["Validation"][i] for i in range(num_classes)]
    test_vals = [counts["Test"][i] for i in range(num_classes)]

    # Konfigurasi Tampilan Grafik (Gaya Publikasi Ilmiah)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)

    color_train = "#1f77b4"  # Biru tua
    color_val = "#ff7f0e"    # Oranye / Amber
    color_test = "#2ca02c"   # Hijau

    rects1 = ax.barh(y_indices + bar_height, train_vals, bar_height, label="Train", color=color_train, alpha=0.9)
    rects2 = ax.barh(y_indices, val_vals, bar_height, label="Validation", color=color_val, alpha=0.9)
    rects3 = ax.barh(y_indices - bar_height, test_vals, bar_height, label="Test", color=color_test, alpha=0.9)

    ax.set_xlabel("Jumlah Objek / Anotasi (Instances)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Kelas Alfabet BISINDO (A - Z)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Distribusi Anotasi Kelas Alfabet BISINDO per Split Data (YOLOv11)", fontsize=13, fontweight="bold", pad=15)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # Agar huruf 'A' berada di posisi paling atas
    ax.legend(loc="lower right", frameon=True, fontsize=11, shadow=True)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Simpan Hasil
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()

    total_instances = sum(train_vals) + sum(val_vals) + sum(test_vals)
    print(f"[SUKSES] Grafik sebaran {num_classes} kelas berhasil disimpan ke: {out_p}")
    print(f"         Total objek terpetakan: {total_instances:,} instances (Train: {sum(train_vals):,}, Val: {sum(val_vals):,}, Test: {sum(test_vals):,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisasi grafik distribusi kelas dataset BISINDO.")
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/bisindo-dataset-1/data.yaml",
        help="Path ke data.yaml (default: datasets/bisindo-dataset-1/data.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/class_distribution.png",
        help="Path tujuan file gambar output (default: results/figures/class_distribution.png)",
    )
    args = parser.parse_args()

    plot_class_distribution(data_yaml_path=args.data, output_path=args.output)
