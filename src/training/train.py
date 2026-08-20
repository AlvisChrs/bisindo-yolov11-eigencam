"""
src/training/train.py
─────────────────────
Training YOLOv11-nano untuk deteksi alfabet BISINDO A-Z.

PERBAIKAN vs notebook lama
--------------------------
- Satu pemanggilan model.train() utuh (40 epoch), bukan dua sesi terpisah.
  Training dua sesi (27 + 13 epoch) menyebabkan LR scheduler cosine annealing
  restart dari awal di sesi kedua, sehingga kurva LR tidak konsisten.
- Resume yang benar menggunakan model.train(resume=True) bawaan Ultralytics,
  bukan YOLO("last.pt").train(epochs=13) yang membuat run baru.
- Auto-detect resume: kalau last.pt sudah ada di folder run (sisa sesi Colab
  sebelumnya yang terputus), script otomatis resume tanpa perlu --resume manual.
- Semua hyperparameter dibaca dari configs/train_config.yaml, tidak ada
  angka hardcoded di file ini.

Penggunaan di Google Colab (Colab free tier)
--------------------------------------------
Karena Colab free tier bisa disconnect sewaktu-waktu, alurnya adalah:

    Sesi pertama — mulai training baru:
        python -m src.training.train

    Sesi berikutnya — Colab putus, mount Drive lagi lalu jalankan perintah SAMA:
        python -m src.training.train

    Script otomatis mendeteksi last.pt ada, langsung resume — kurva LR tetap
    nyambung dari posisi terakhir. Tidak perlu tambahkan --resume manual.

    Atau eksplisit pakai --resume kalau mau yakin:
        python -m src.training.train --resume

Penggunaan lain
---------------
    # Dengan config eksplisit:
    python -m src.training.train --config configs/train_config.yaml

    # Override data path:
    python -m src.training.train --data datasets/bisindo-dataset-1/data.yaml

    # Sanity check 1 epoch:
    python -m src.training.train --epochs 1 --name bisindo_sanity

    # Training tahan-banting (simpan checkpoint tiap epoch, aman buat Colab free):
    python -m src.training.train --save-period 1
"""

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """
    Baca train_config.yaml dan kembalikan sebagai dict.

    Raises
    ------
    SystemExit
        Jika file tidak ditemukan atau YAML tidak valid.
    """
    path = Path(config_path)
    if not path.exists():
        print(
            f"[ERROR] File konfigurasi tidak ditemukan: {path.resolve()}\n"
            "Pastikan kamu menjalankan perintah dari root project.",
            file=sys.stderr,
        )
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"[ERROR] Gagal membaca {path}: {e}", file=sys.stderr)
            sys.exit(1)

    return cfg


def validate_data_yaml(data_path: str) -> None:
    """
    Pastikan file data.yaml ada sebelum training dimulai.

    Raises
    ------
    SystemExit
        Jika file tidak ditemukan.
    """
    p = Path(data_path)
    if not p.exists():
        print(
            f"[ERROR] data.yaml tidak ditemukan: {p.resolve()}\n"
            "Jalankan dulu: python -m src.data.download_dataset",
            file=sys.stderr,
        )
        sys.exit(1)


def find_last_checkpoint(project: str, name: str) -> Path | None:
    """
    Cari last.pt dari run sebelumnya (sisa sesi Colab yang terputus).

    Parameters
    ----------
    project : str
        Folder project training (misal: "results" atau path Google Drive).
    name : str
        Nama run (misal: "bisindo_yolo11n").

    Returns
    -------
    Path | None
        Path ke last.pt jika ada, None jika tidak ada (artinya training baru).
    """
    last_pt = Path(project) / name / "weights" / "last.pt"
    if last_pt.exists():
        return last_pt
    return None


def train(config_path: str = "configs/train_config.yaml", overrides: dict = None) -> str:
    """
    Jalankan training YOLOv11-nano — otomatis resume jika last.pt ada.

    Perilaku auto-detect:
    - Jika last.pt TIDAK ada di folder run → mulai training baru dari yolo11n.pt
    - Jika last.pt ADA di folder run → resume dari checkpoint terakhir
      (LR scheduler melanjutkan kurva yang sama, bukan restart)

    Ini membuat perintah `python -m src.training.train` aman dijalankan
    berulang kali di Colab tanpa harus ingat apakah ini sesi pertama atau
    lanjutan — script yang akan memutuskan sendiri.

    Parameters
    ----------
    config_path : str
        Path ke train_config.yaml.
    overrides : dict, optional
        Override nilai dari CLI. None-valued keys diabaikan.

    Returns
    -------
    str
        Path ke folder hasil training.
    """
    cfg = load_config(config_path)

    # Terapkan overrides dari CLI (skip yang None)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    project = cfg.get("project", "results")
    name    = cfg.get("name", "bisindo_yolo11n")

    # ── Auto-detect: resume atau mulai baru? ─────────────────────────────────
    last_pt = find_last_checkpoint(project, name)

    if last_pt is not None:
        print(
            f"\n[INFO] Ditemukan checkpoint dari sesi sebelumnya: {last_pt}\n"
            f"       → Otomatis RESUME (LR scheduler melanjutkan kurva yang sama)\n"
            f"         Jika ini tidak diinginkan, hapus folder: {Path(project) / name}\n"
        )
        model = YOLO(str(last_pt))
        results = model.train(resume=True)

    else:
        # Validasi dataset hanya saat mulai baru
        validate_data_yaml(cfg["data"])

        print(
            f"\n[INFO] Tidak ada checkpoint sebelumnya.\n"
            f"       → Mulai training baru dari: {cfg['model']}\n"
            f"       Model   : {cfg['model']}\n"
            f"       Data    : {cfg['data']}\n"
            f"       Epochs  : {cfg['epochs']} (target total)\n"
            f"       Batch   : {cfg['batch']}\n"
            f"       Imgsz   : {cfg['imgsz']}\n"
            f"       Device  : {cfg.get('device', '') or 'auto'}\n"
            f"       Output  : {project}/{name}\n"
            f"\n[TIP] Jika Colab disconnect, jalankan perintah yang sama lagi —\n"
            f"      script otomatis melanjutkan dari epoch terakhir.\n"
        )

        model = YOLO(cfg["model"])

        # Semua hyperparameter dari cfg — tidak ada angka hardcoded di sini
        train_args = {
            "data":         cfg["data"],
            "epochs":       cfg["epochs"],
            "batch":        cfg["batch"],
            "imgsz":        cfg["imgsz"],
            "optimizer":    cfg["optimizer"],
            "lr0":          cfg["lr0"],
            "lrf":          cfg.get("lrf", 0.01),
            "momentum":     cfg.get("momentum", 0.937),
            "weight_decay": cfg.get("weight_decay", 0.0005),
            "conf":         cfg["conf"],
            "iou":          cfg["iou"],
            "device":       cfg.get("device", ""),
            "workers":      cfg.get("workers", 4),
            "project":      project,
            "name":         name,
            "exist_ok":     cfg.get("exist_ok", False),
            "save":         cfg.get("save", True),
            "save_period":  cfg.get("save_period", -1),
            "plots":        cfg.get("plots", True),
            "verbose":      cfg.get("verbose", True),
            # Augmentasi
            "hsv_h":    cfg.get("hsv_h", 0.015),
            "hsv_s":    cfg.get("hsv_s", 0.7),
            "hsv_v":    cfg.get("hsv_v", 0.4),
            "degrees":  cfg.get("degrees", 0.0),
            "translate":cfg.get("translate", 0.1),
            "scale":    cfg.get("scale", 0.5),
            "shear":    cfg.get("shear", 0.0),
            "flipud":   cfg.get("flipud", 0.0),
            "fliplr":   cfg.get("fliplr", 0.5),
            "mosaic":   cfg.get("mosaic", 1.0),
            "mixup":    cfg.get("mixup", 0.0),
        }

        results = model.train(**train_args)

    # Ambil path output
    output_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name

    print(f"\n[DONE] Training selesai / sesi ini berakhir.")
    print(f"       Hasil tersimpan di : {output_dir.resolve()}")
    print(f"       best.pt            : {output_dir / 'weights' / 'best.pt'}")
    print(f"       last.pt            : {output_dir / 'weights' / 'last.pt'}")

    return str(output_dir)


def main() -> None:
    """Entry point untuk penggunaan via CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Training YOLOv11-nano BISINDO A-Z.\n"
            "Auto-detect resume: kalau last.pt sudah ada, otomatis lanjut dari sana.\n"
            "Aman dijalankan berulang kali di Colab tanpa argumen tambahan."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path ke file konfigurasi YAML",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Paksa resume dari last.pt (berguna kalau folder run ada tapi "
            "auto-detect tidak mendeteksi, misal path project berbeda di Colab)"
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override path data.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override jumlah epoch (misal: 1 untuk sanity check)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override nama run output",
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Override folder project. Di Colab pakai path Drive:\n"
            "  --project /content/drive/MyDrive/Project_BISINDO"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device: '0' (GPU), 'cpu'. Default: auto",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader workers. Set 0 jika error di Windows",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=None,
        dest="save_period",
        help=(
            "Simpan checkpoint tiap N epoch. "
            "Gunakan --save-period 1 di Colab free untuk tahan-banting disconnect. "
            "Default: -1 (hanya simpan best.pt dan last.pt)"
        ),
    )
    args = parser.parse_args()

    # Kalau --resume dipakai eksplisit, bypass auto-detect dan langsung resume
    if args.resume:
        cfg = load_config(args.config)
        project = args.project or cfg.get("project", "results")
        name    = args.name    or cfg.get("name", "bisindo_yolo11n")
        last_pt = Path(project) / name / "weights" / "last.pt"

        if not last_pt.exists():
            print(
                f"[ERROR] --resume dipakai tapi last.pt tidak ditemukan: {last_pt.resolve()}\n"
                "Belum ada training sebelumnya yang bisa dilanjutkan.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[INFO] Resume eksplisit dari: {last_pt.resolve()}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
        output_dir = Path(results.save_dir) if hasattr(results, "save_dir") else last_pt.parent.parent
        print(f"\n[DONE] Resume selesai. Hasil di: {output_dir.resolve()}")

    else:
        overrides = {
            "data":        args.data,
            "epochs":      args.epochs,
            "name":        args.name,
            "project":     args.project,
            "device":      args.device,
            "workers":     args.workers,
            "save_period": args.save_period,
        }
        train(config_path=args.config, overrides=overrides)


if __name__ == "__main__":
    main()
