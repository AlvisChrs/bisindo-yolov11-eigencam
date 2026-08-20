"""
src/inference/predict.py
────────────────────────
Inference YOLOv11-nano untuk deteksi alfabet BISINDO A-Z.

Mendukung dua mode input:
- Satu gambar (path ke file .jpg/.png)
- Satu folder (semua gambar di dalamnya diproses sekaligus)

Output
------
- Gambar dengan bounding box dan label disimpan ke folder --output
- Ringkasan prediksi dicetak ke terminal
- Opsional: simpan hasil dalam predictions.json

Penggunaan
----------
    # Satu gambar:
    python -m src.inference.predict \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/A_001.jpg

    # Seluruh folder test:
    python -m src.inference.predict \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/

    # Dengan threshold berbeda:
    python -m src.inference.predict \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/ \\
        --conf    0.3

    # Simpan JSON ringkasan:
    python -m src.inference.predict \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/ \\
        --save-json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# Format gambar yang didukung
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_image_paths(source: str) -> list[Path]:
    """
    Kumpulkan semua path gambar dari source (file tunggal atau folder).

    Parameters
    ----------
    source : str
        Path ke satu file gambar atau folder berisi gambar.

    Returns
    -------
    list[Path]
        Daftar path gambar yang ditemukan.

    Raises
    ------
    SystemExit
        Jika source tidak ditemukan atau tidak ada gambar.
    """
    src = Path(source)
    if not src.exists():
        print(f"[ERROR] Source tidak ditemukan: {src.resolve()}", file=sys.stderr)
        sys.exit(1)

    if src.is_file():
        if src.suffix.lower() not in IMG_EXTENSIONS:
            print(f"[ERROR] File bukan gambar: {src}", file=sys.stderr)
            sys.exit(1)
        return [src]

    # Folder — kumpulkan semua gambar
    images = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    if not images:
        print(f"[ERROR] Tidak ada gambar ditemukan di folder: {src.resolve()}", file=sys.stderr)
        sys.exit(1)

    return images


def predict_single(
    model,
    img_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> dict:
    """
    Jalankan inference pada satu gambar menggunakan model.predict() Ultralytics.

    Ultralytics secara internal sudah menggunakan LetterBox preprocessing
    saat model.predict() dipanggil — gambar di-resize + pad ke imgsz×imgsz
    tanpa distorsi aspek rasio. Ini berbeda dari notebook lama yang pakai
    cv2.resize() polos dan bisa mendistorsi objek tangan.

    Parameters
    ----------
    model : YOLO
        Model yang sudah dimuat.
    img_path : Path
        Path ke file gambar.
    conf : float
        Confidence threshold.
    iou : float
        IoU threshold untuk NMS.
    imgsz : int
        Ukuran input model.
    device : str
        Device komputasi.

    Returns
    -------
    dict
        Hasil prediksi: filename, detections (list bounding box + label + conf).
    """
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
    )

    result = results[0]
    detections = []

    if len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_val = float(box.conf[0])

            detections.append({
                "label":      cls_name,
                "confidence": round(conf_val, 4),
                "bbox_xyxy":  [x1, y1, x2, y2],
            })

    return {
        "filename":   img_path.name,
        "path":       str(img_path),
        "detections": detections,
        "n_detections": len(detections),
    }


def save_annotated_image(
    model,
    img_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    output_dir: Path,
) -> Path:
    """
    Simpan gambar dengan bounding box dan label ke output_dir.

    Menggunakan result.plot() dari Ultralytics yang sudah menangani
    koordinat bounding box dengan benar (konsisten dengan LetterBox).

    Parameters
    ----------
    model : YOLO
        Model yang sudah dimuat.
    img_path : Path
        Path ke file gambar input.
    conf, iou, imgsz, device : float/int/str
        Parameter inference.
    output_dir : Path
        Folder tujuan penyimpanan gambar hasil.

    Returns
    -------
    Path
        Path ke file gambar hasil yang disimpan.
    """
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
    )

    result = results[0]
    annotated_bgr = result.plot()  # gambar BGR dengan bounding box

    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), annotated_bgr)

    return out_path


def predict(
    weights: str,
    source: str,
    conf: float = 0.5,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "",
    output: str = "results/predictions",
    save_json: bool = False,
) -> list[dict]:
    """
    Jalankan inference pada satu gambar atau seluruh folder.

    Parameters
    ----------
    weights : str
        Path ke file .pt model.
    source : str
        Path ke gambar tunggal atau folder berisi gambar.
    conf : float
        Confidence threshold. Default: 0.5.
    iou : float
        IoU threshold untuk NMS. Default: 0.7.
    imgsz : int
        Ukuran input. Default: 640.
    device : str
        Device komputasi. Default: "" (auto).
    output : str
        Folder tujuan gambar hasil annotasi. Default: "results/predictions".
    save_json : bool
        Simpan predictions.json ke output folder. Default: False.

    Returns
    -------
    list[dict]
        List hasil prediksi tiap gambar.

    Raises
    ------
    SystemExit
        Jika weights atau source tidak ditemukan.
    """
    from ultralytics import YOLO

    # Validasi weights
    weights_path = Path(weights)
    if not weights_path.exists():
        print(
            f"[ERROR] Weights tidak ditemukan: {weights_path.resolve()}\n"
            "Jalankan training dulu: python -m src.training.train",
            file=sys.stderr,
        )
        sys.exit(1)

    # Kumpulkan gambar
    image_paths = collect_image_paths(source)

    # Siapkan output dir
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model   : {weights_path}")
    print(f"[INFO] Source  : {source}  ({len(image_paths)} gambar)")
    print(f"[INFO] Conf    : {conf}  |  IoU: {iou}  |  Imgsz: {imgsz}")
    print(f"[INFO] Output  : {output_dir.resolve()}")
    print()

    model = YOLO(str(weights_path))
    all_results = []
    detected_count = 0

    for i, img_path in enumerate(image_paths, start=1):
        # Prediksi + simpan gambar annotasi
        pred = predict_single(model, img_path, conf, iou, imgsz, device)
        save_annotated_image(model, img_path, conf, iou, imgsz, device, output_dir)

        all_results.append(pred)
        detected_count += pred["n_detections"]

        # Tampilkan ringkasan per gambar
        if pred["n_detections"] > 0:
            labels = [f"{d['label']}({d['confidence']:.2f})" for d in pred["detections"]]
            print(f"  [{i:>4}/{len(image_paths)}] {img_path.name:<30} → {', '.join(labels)}")
        else:
            print(f"  [{i:>4}/{len(image_paths)}] {img_path.name:<30} → (tidak terdeteksi)")

    # Simpan JSON jika diminta
    if save_json:
        json_path = output_dir / "predictions.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Hasil prediksi disimpan di: {json_path}")

    print(
        f"\n[DONE] Selesai memproses {len(image_paths)} gambar.\n"
        f"       Total deteksi : {detected_count}\n"
        f"       Gambar output : {output_dir.resolve()}"
    )

    return all_results


def main() -> None:
    """Entry point untuk penggunaan via CLI."""
    parser = argparse.ArgumentParser(
        description="Inference YOLOv11-nano BISINDO — satu gambar atau folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path ke file .pt model",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path ke gambar tunggal atau folder berisi gambar",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold untuk NMS",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran gambar input",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device: '0' (GPU), 'cpu'. Default: auto",
    )
    parser.add_argument(
        "--output",
        default="results/predictions",
        help="Folder output gambar dengan bounding box",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Simpan hasil prediksi sebagai predictions.json",
    )
    args = parser.parse_args()

    predict(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        output=args.output,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
