"""
src/data/download_dataset.py
────────────────────────────
Mengunduh dataset BISINDO A-Z dari Roboflow Universe.

Dataset : bisindo/bisindo-dataset versi 1
Lisensi : CC BY 4.0 — https://universe.roboflow.com/bisindo/bisindo-dataset
Format  : yolov11

Penggunaan
----------
    # Via CLI (dari root project):
    python -m src.data.download_dataset

    # Dengan argumen opsional:
    python -m src.data.download_dataset --workspace bisindo --project bisindo-dataset --version 1 --dest datasets/

    # Dari kode Python lain:
    from src.data.download_dataset import download_dataset
    dataset_path = download_dataset()

Prasyarat
---------
    1. Buat file .env di root project (copy dari .env.example)
    2. Isi ROBOFLOW_API_KEY dengan API key Roboflow kamu
       (buka https://app.roboflow.com → Settings → API Keys)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


def _load_api_key() -> str:
    """
    Muat ROBOFLOW_API_KEY dari file .env (root project) atau environment.

    Raises
    ------
    SystemExit
        Jika ROBOFLOW_API_KEY tidak ditemukan, program berhenti dengan
        pesan error yang jelas — bukan traceback mentah.
    """
    # Cari .env dari root project (dua level atas dari file ini: src/data/ → root)
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Fallback: coba load dari working directory
        load_dotenv()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    if not api_key:
        print(
            "\n[ERROR] ROBOFLOW_API_KEY tidak ditemukan.\n"
            "\nLangkah perbaikan:\n"
            f"  1. Salin .env.example menjadi .env di folder: {project_root}\n"
            "  2. Buka https://app.roboflow.com → Settings → API Keys\n"
            "  3. Tempel API key kamu ke dalam .env:\n"
            "         ROBOFLOW_API_KEY=your_key_here\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_key


def download_dataset(
    workspace: str = "bisindo",
    project: str = "bisindo-dataset",
    version: int = 1,
    dest: str = "datasets",
) -> str:
    """
    Unduh dataset dari Roboflow dan kembalikan path folder hasilnya.

    Parameters
    ----------
    workspace : str
        Nama workspace Roboflow. Default: "bisindo".
    project : str
        Nama project Roboflow. Default: "bisindo-dataset".
    version : int
        Nomor versi dataset. Default: 1.
    dest : str
        Folder tujuan unduhan (relatif terhadap working directory).
        Default: "datasets".

    Returns
    -------
    str
        Path absolut ke folder dataset yang sudah diunduh.
        Berisi sub-folder train/, valid/, test/ dan file data.yaml.
        Selalu mengembalikan path ke subfolder `bisindo-dataset-1/`.

    Raises
    ------
    SystemExit
        Jika API key tidak ditemukan (lihat _load_api_key).
    Exception
        Jika ada error dari Roboflow (koneksi gagal, versi tidak ada, dll).
    """
    api_key = _load_api_key()

    # Tentukan path tujuan absolut: selalu gunakan subfolder bisindo-dataset-1
    project_root = Path(__file__).resolve().parents[2]
    dest_path = (project_root / dest).resolve()
    # Subfolder tetap untuk konsistensi (match configs/train_config.yaml)
    dataset_subdir = dest_path / "bisindo-dataset-1"
    dataset_subdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Menghubungi Roboflow... workspace={workspace}, project={project}, version={version}")
    print(f"[INFO] Dataset akan disimpan di: {dataset_subdir}")

    # Cek apakah dataset sudah ada di subfolder yang benar
    existing_data_yaml = dataset_subdir / "data.yaml"
    if existing_data_yaml.exists():
        print(f"[INFO] Dataset sudah ada, skip download. data.yaml: {existing_data_yaml}")
        return str(dataset_subdir)

    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_version = rf_project.version(version)

    # Download ke folder parent (dest_path), SDK akan buat folder sendiri
    # overwrite=True supaya SDK tidak skip karena folder sudah ada
    dataset = rf_version.download(
        model_format="yolov11",
        location=str(dest_path),
        overwrite=True,
    )

    # Cari data.yaml — SDK kadang menaruhnya di subfolder, kadang langsung di dest_path
    dataset_location = Path(dataset.location).resolve()
    data_yaml = dataset_location / "data.yaml"

    if not data_yaml.exists():
        candidates = list(dest_path.rglob("data.yaml"))
        if candidates:
            dataset_location = candidates[0].parent
            data_yaml = candidates[0]
        else:
            print(
                f"[WARNING] data.yaml tidak ditemukan di {dest_path}.\n"
                "Kemungkinan download gagal di tengah jalan. Coba jalankan ulang.",
                file=sys.stderr,
            )
            return str(dataset_subdir)

    # Jika dataset_location BUKAN dataset_subdir, pindahkan isinya ke dataset_subdir
    if dataset_location.resolve() != dataset_subdir.resolve():
        print(f"[INFO] Memindahkan dataset dari {dataset_location} ke {dataset_subdir}...")
        import shutil
        # Hapus dataset_subdir kosong kalau ada
        if dataset_subdir.exists():
            shutil.rmtree(dataset_subdir)
        dataset_subdir.mkdir(parents=True, exist_ok=True)
        # Pindahkan KONTEN folder hasil download ke dataset_subdir (bukan folder-nya)
        for item in dataset_location.iterdir():
            shutil.move(str(item), str(dataset_subdir / item.name))
        data_yaml = dataset_subdir / "data.yaml"

    print(f"[OK] Dataset siap. data.yaml: {data_yaml}")
    return str(dataset_subdir)


def main() -> None:
    """Entry point untuk penggunaan via CLI."""
    parser = argparse.ArgumentParser(
        description="Download dataset BISINDO dari Roboflow Universe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workspace",
        default="bisindo",
        help="Nama workspace Roboflow",
    )
    parser.add_argument(
        "--project",
        default="bisindo-dataset",
        help="Nama project Roboflow",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Nomor versi dataset",
    )
    parser.add_argument(
        "--dest",
        default="datasets",
        help="Folder tujuan unduhan (relatif dari root project)",
    )
    args = parser.parse_args()

    location = download_dataset(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        dest=args.dest,
    )
    print(f"\n[DONE] Dataset tersimpan di: {location}")


if __name__ == "__main__":
    main()
