"""
src/data/visualize_augmentation.py
────────────────────────────────────────────────────────────────────────────
Membuat grid visualisasi teknik-teknik augmentasi yang digunakan pada
training YOLOv11-nano BISINDO, sesuai parameter di configs/train_config.yaml.

Augmentasi yang divisualisasikan:
  1. Original (+ bounding box)
  2. Horizontal Flip          — fliplr = 0.5
  3. HSV Color Augmentation   — hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
  4. Mosaic                   — mosaic = 1.0  (4 gambar digabung)
  5. Random Erasing           — erasing = 0.4

Output:
  results/figures/augmentation_preview.png  (dpi 200+, siap laporan skripsi)

Penggunaan:
  # Dari root project:
  python -m src.data.visualize_augmentation

  # Dengan argumen opsional:
  python -m src.data.visualize_augmentation --seed 42 --sample-idx 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Konstanta / konfigurasi ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_IMG_DIR = PROJECT_ROOT / "datasets" / "bisindo-dataset-1" / "train" / "images"
TRAIN_LBL_DIR = PROJECT_ROOT / "datasets" / "bisindo-dataset-1" / "train" / "labels"
OUTPUT_DIR    = PROJECT_ROOT / "results" / "figures"
OUTPUT_FILE   = OUTPUT_DIR   / "augmentation_preview.png"

# Parameter augmentasi — PERSIS sesuai configs/train_config.yaml
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
ERASING_RATIO = 0.4
IMG_SIZE = 640

# Nama kelas BISINDO (26 huruf A-Z)
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

BOX_COLOR_BGR = (0, 200, 255)


# ── Fungsi utilitas ──────────────────────────────────────────────────────────

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Gambar tidak bisa dibuka: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_boxes(img: np.ndarray, boxes: list, thickness: int = 2) -> np.ndarray:
    out_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    for cls_id, x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), BOX_COLOR_BGR, thickness)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ly = max(y1 - th - bl, 0)
        cv2.rectangle(out_bgr, (x1, ly), (x1 + tw, ly + th + bl), BOX_COLOR_BGR, -1)
        cv2.putText(out_bgr, label, (x1, ly + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def resize_square(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    yo, xo = (size - nh) // 2, (size - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def resize_boxes_letterbox(boxes: list, orig_h: int, orig_w: int, size: int = IMG_SIZE) -> list:
    scale = size / max(orig_h, orig_w)
    xo = (size - int(orig_w * scale)) // 2
    yo = (size - int(orig_h * scale)) // 2
    return [(cls, int(x1*scale)+xo, int(y1*scale)+yo, int(x2*scale)+xo, int(y2*scale)+yo)
            for cls, x1, y1, x2, y2 in boxes]


# ── Augmentasi ────────────────────────────────────────────────────────────────

def aug_original(img, boxes):
    return draw_boxes(img, boxes)


def aug_fliplr(img, boxes):
    flipped = np.fliplr(img).copy()
    w = img.shape[1]
    new_boxes = [(cls, w - x2, y1, w - x1, y2) for cls, x1, y1, x2, y2 in boxes]
    return draw_boxes(flipped, new_boxes)


def aug_hsv(img, boxes, rng):
    # Identik dengan ultralytics augment_hsv()
    r = rng.uniform(-1, 1, 3) * [HSV_H, HSV_S, HSV_V] + 1
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hue, sat, val = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    aug_bgr = cv2.cvtColor(
        cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))),
        cv2.COLOR_HSV2BGR,
    )
    return draw_boxes(cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB), boxes)


def aug_mosaic(all_img_paths: list, primary_idx: int, size: int, rng) -> np.ndarray:
    other = [i for i in range(len(all_img_paths)) if i != primary_idx]
    chosen = rng.choice(other, size=3, replace=False).tolist()
    indices = [primary_idx] + chosen

    xc = int(rng.uniform(0.4, 0.6) * size)
    yc = int(rng.uniform(0.4, 0.6) * size)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    placements = [
        (0, (0, 0, xc, yc), "br"),
        (1, (xc, 0, size, yc), "bl"),
        (2, (0, yc, xc, size), "tr"),
        (3, (xc, yc, size, size), "tl"),
    ]

    for qi, (dx1, dy1, dx2, dy2), corner in placements:
        tile = load_image_rgb(all_img_paths[indices[qi]])
        th, tw = tile.shape[:2]
        dw, dh = dx2 - dx1, dy2 - dy1
        scale = max(dw / tw, dh / th)
        ntw, nth = int(tw * scale), int(th * scale)
        tile = cv2.resize(tile, (ntw, nth), interpolation=cv2.INTER_LINEAR)
        sx = max(0, ntw - dw) if corner in ("br", "tr") else 0
        sy = max(0, nth - dh) if corner in ("br", "bl") else 0
        crop = tile[sy:sy + dh, sx:sx + dw]
        ch, cw = crop.shape[:2]
        canvas[dy1:dy1 + ch, dx1:dx1 + cw] = crop

    return canvas


def aug_erasing(img, boxes, rng):
    result = img.copy()
    h, w = img.shape[:2]
    area = h * w
    for _ in range(10):
        target = rng.uniform(0.02, ERASING_RATIO) * area
        asp = rng.uniform(0.3, 3.3)
        eh = int(round(np.sqrt(target * asp)))
        ew = int(round(np.sqrt(target / asp)))
        if eh >= h or ew >= w:
            continue
        x0 = int(rng.integers(0, w - ew))
        y0 = int(rng.integers(0, h - eh))
        result[y0:y0 + eh, x0:x0 + ew] = rng.integers(0, 256, (eh, ew, 3), dtype=np.uint8)
        break
    return draw_boxes(result, boxes)


# ── Grid ──────────────────────────────────────────────────────────────────────

def build_grid(panels: list, ncols: int = 3, dpi: int = 220):
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    cell = 3.5
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell * ncols, cell * nrows + 0.6), dpi=dpi)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    fig.patch.set_facecolor("#1a1a2e")
    border_colors = ["#00b4d8", "#48cae4", "#90e0ef", "#caf0f8", "#ade8f4", "#023e8a"]

    for idx, (title, img) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(title, fontsize=9.5, color="#e0e0e0", fontweight="bold", pad=5)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_colors[idx % len(border_colors)])
            spine.set_linewidth(2)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.set_facecolor("#12121f")
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#2a2a4a")
            sp.set_linewidth(1)

    fig.suptitle("Visualisasi Teknik Augmentasi Data — BISINDO YOLOv11-nano",
                 fontsize=13, color="#ffffff", fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(sample_idx: int = 0, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    img_paths = (sorted(TRAIN_IMG_DIR.glob("*.jpg")) +
                 sorted(TRAIN_IMG_DIR.glob("*.jpeg")) +
                 sorted(TRAIN_IMG_DIR.glob("*.png")))

    if not img_paths:
        print(f"[ERROR] Tidak ada gambar di {TRAIN_IMG_DIR}.\n"
              "Pastikan dataset sudah didownload: python -m src.data.download_dataset",
              file=sys.stderr)
        sys.exit(1)

    sample_idx = sample_idx % len(img_paths)
    img_path = img_paths[sample_idx]
    print(f"[INFO] Gambar sampel: {img_path.name}  (indeks {sample_idx})")

    img_rgb = load_image_rgb(img_path)
    orig_h, orig_w = img_rgb.shape[:2]
    lbl_path = TRAIN_LBL_DIR / img_path.with_suffix(".txt").name
    boxes_px = load_yolo_labels(lbl_path, orig_w, orig_h)

    img_sq = resize_square(img_rgb, IMG_SIZE)
    boxes_sq = resize_boxes_letterbox(boxes_px, orig_h, orig_w, IMG_SIZE)
    print(f"[INFO] Ukuran resize: {IMG_SIZE}x{IMG_SIZE}, label: {len(boxes_sq)} box")

    print("[INFO] Menerapkan augmentasi...")
    panels = [
        ("(1) Original\n(+ Bounding Box)",           aug_original(img_sq, boxes_sq)),
        ("(2) Horizontal Flip\n[fliplr = 0.5]",      aug_fliplr(img_sq, boxes_sq)),
        ("(3) HSV Augmentation\n[h=0.015  s=0.7  v=0.4]", aug_hsv(img_sq, boxes_sq, rng)),
        ("(4) Mosaic (4 gambar)\n[mosaic = 1.0]",    aug_mosaic(img_paths, sample_idx, IMG_SIZE, rng)),
        ("(5) Random Erasing\n[erasing = 0.4]",      aug_erasing(img_sq, boxes_sq, rng)),
    ]

    fig = build_grid(panels, ncols=3, dpi=220)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK]   Grid augmentasi tersimpan -> {OUTPUT_FILE}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualisasi teknik augmentasi data BISINDO YOLOv11-nano.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Indeks gambar sampel dari folder train/images/ (0-based).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed untuk reproduksibilitas.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(sample_idx=args.sample_idx, seed=args.seed)
