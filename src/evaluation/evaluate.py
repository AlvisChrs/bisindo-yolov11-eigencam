"""
src/evaluation/evaluate.py
──────────────────────────
Evaluasi model YOLOv11-nano hasil training pada test set BISINDO A-Z.

Metrik yang dihitung
--------------------
- mAP@50        : mean Average Precision pada IoU threshold 0.50
- mAP@50-95     : mean Average Precision rata-rata IoU 0.50–0.95
- Precision      : rata-rata per kelas
- Recall         : rata-rata per kelas
- Per-class AP   : Average Precision tiap huruf A–Z
- Confusion matrix (disimpan sebagai gambar PNG)

Output
------
Semua hasil disimpan di folder --output (default: results/evaluation/).
- metrics.json     : ringkasan metrik numerik
- metrics.txt      : ringkasan teks yang bisa langsung dikutip di skripsi
- confusion_matrix.png  : dari Ultralytics (jika plots=True)

Penggunaan
----------
    # Evaluasi dengan config default:
    python -m src.evaluation.evaluate --weights results/bisindo_yolo11n/weights/best.pt

    # Dengan config eksplisit:
    python -m src.evaluation.evaluate \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --data    datasets/bisindo-dataset-1/data.yaml \\
        --split   test \\
        --output  results/evaluation
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_data_path_from_config(config_path: str) -> str:
    """Baca path data.yaml dari train_config.yaml sebagai fallback."""
    p = Path(config_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("data")


def evaluate(
    weights: str,
    data: str = None,
    split: str = "test",
    conf: float = 0.5,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "",
    output: str = "results/evaluation",
    config_path: str = "configs/train_config.yaml",
) -> dict:
    """
    Evaluasi model pada test/val set dan simpan metrik ke disk.

    Parameters
    ----------
    weights : str
        Path ke file .pt model (best.pt atau last.pt).
    data : str, optional
        Path ke data.yaml. Jika None, dibaca dari train_config.yaml.
    split : str
        Split yang dievaluasi: "test", "val", atau "train". Default: "test".
    conf : float
        Confidence threshold. Default: 0.5 (sesuai proposal).
    iou : float
        IoU threshold untuk NMS. Default: 0.7.
    imgsz : int
        Ukuran gambar inferensi. Default: 640.
    device : str
        Device komputasi. Default: "" (auto).
    output : str
        Folder untuk menyimpan hasil. Default: "results/evaluation".
    config_path : str
        Path ke train_config.yaml (fallback untuk data path).

    Returns
    -------
    dict
        Dictionary berisi metrik: map50, map50_95, precision, recall,
        dan per_class (dict huruf → AP).

    Raises
    ------
    SystemExit
        Jika weights atau data.yaml tidak ditemukan.
    """
    # Lazy import agar file bisa di-import tanpa ultralytics saat testing
    from ultralytics import YOLO

    # Validasi weights
    weights_path = Path(weights)
    if not weights_path.exists():
        print(
            f"[ERROR] File weights tidak ditemukan: {weights_path.resolve()}\n"
            "Jalankan training dulu: python -m src.training.train",
            file=sys.stderr,
        )
        sys.exit(1)

    # Tentukan data.yaml
    if data is None:
        data = load_data_path_from_config(config_path)
    if data is None or not Path(data).exists():
        print(
            f"[ERROR] data.yaml tidak ditemukan: {data}\n"
            "Jalankan dulu: python -m src.data.download_dataset",
            file=sys.stderr,
        )
        sys.exit(1)

    # Siapkan folder output
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Memuat model: {weights_path}")
    print(f"[INFO] Dataset     : {data}")
    print(f"[INFO] Split       : {split}")
    print(f"[INFO] Output      : {output_dir.resolve()}")
    print()

    model = YOLO(str(weights_path))

    # Jalankan evaluasi via model.val()
    # Ultralytics otomatis menghitung mAP, P, R, confusion matrix
    val_results = model.val(
        data=str(data),
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        plots=True,         # simpan confusion matrix & kurva P-R
        save_json=False,
        verbose=True,
        project=str(output_dir),
        name="val_run",
        exist_ok=True,
    )

    # Ekstrak metrik dari hasil validasi
    # val_results.box berisi DetMetrics untuk object detection
    box = val_results.box

    map50     = float(box.map50)       if hasattr(box, "map50")     else float(box.map)
    map50_95  = float(box.map)         if hasattr(box, "map")       else 0.0
    precision = float(box.mp)          if hasattr(box, "mp")        else 0.0
    recall    = float(box.mr)          if hasattr(box, "mr")        else 0.0

    # Per-class AP (AP@50 per huruf)
    class_names = val_results.names   # dict {idx: nama_kelas}
    per_class = {}
    if hasattr(box, "ap_class_index") and hasattr(box, "ap50"):
        for idx, ap_val in zip(box.ap_class_index, box.ap50):
            cls_name = class_names.get(int(idx), str(idx))
            per_class[cls_name] = round(float(ap_val), 4)

    metrics = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "weights":      str(weights_path.resolve()),
        "data":         str(data),
        "split":        split,
        "conf":         conf,
        "iou":          iou,
        "imgsz":        imgsz,
        "map50":        round(map50, 4),
        "map50_95":     round(map50_95, 4),
        "precision":    round(precision, 4),
        "recall":       round(recall, 4),
        "per_class_ap": per_class,
    }

    # Simpan metrics.json
    json_path = output_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Simpan metrics.txt (format ringkasan skripsi)
    txt_path = output_dir / "metrics.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("HASIL EVALUASI — BISINDO YOLOv11-nano\n")
        f.write("=" * 50 + "\n")
        f.write(f"Waktu      : {metrics['timestamp']}\n")
        f.write(f"Weights    : {metrics['weights']}\n")
        f.write(f"Dataset    : {metrics['data']}\n")
        f.write(f"Split      : {metrics['split']}\n")
        f.write(f"Conf thr   : {metrics['conf']}\n")
        f.write(f"IoU thr    : {metrics['iou']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"mAP@50     : {metrics['map50']:.4f}  ({metrics['map50']*100:.2f}%)\n")
        f.write(f"mAP@50-95  : {metrics['map50_95']:.4f}  ({metrics['map50_95']*100:.2f}%)\n")
        f.write(f"Precision  : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)\n")
        f.write(f"Recall     : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)\n")
        f.write("-" * 50 + "\n")
        if per_class:
            f.write("Per-class AP@50:\n")
            for cls_name in sorted(per_class.keys()):
                f.write(f"  {cls_name:<4}: {per_class[cls_name]:.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"\n[DONE] Evaluasi selesai.")
    print(f"       mAP@50    : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"       mAP@50-95 : {map50_95:.4f}  ({map50_95*100:.2f}%)")
    print(f"       Precision : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"       Recall    : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"\n       Tersimpan di: {output_dir.resolve()}")
    print(f"       - metrics.json")
    print(f"       - metrics.txt")

    return metrics


def main() -> None:
    """Entry point untuk penggunaan via CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluasi model YOLOv11-nano BISINDO — hitung mAP, P, R, confusion matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path ke file .pt model (misal: results/bisindo_yolo11n/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path ke data.yaml. Jika tidak diisi, dibaca dari train_config.yaml",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train"],
        help="Split dataset yang dievaluasi",
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
        help="Ukuran gambar",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device komputasi: '0' (GPU), 'cpu', dll. Default: auto",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation",
        help="Folder output untuk menyimpan metrics.json dan metrics.txt",
    )
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path ke train_config.yaml (untuk fallback data path)",
    )
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        output=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
