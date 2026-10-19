"""
src/utils/check_env.py
──────────────────────
Utilitas diagnostik untuk memeriksa kesiapan lingkungan eksekusi (hardware,
driver CUDA/GPU, PyTorch, dan pustaka pendukung YOLOv11/Eigen-CAM).

Penggunaan:
-----------
    python -m src.utils.check_env
"""

import os
import platform
import sys


def get_size_format(b: float, factor: int = 1024, suffix: str = "B") -> str:
    """Format bytes ke satuan yang mudah dibaca (MB, GB)."""
    for unit in ["", "K", "M", "G", "T"]:
        if b < factor:
            return f"{b:.2f} {unit}{suffix}"
        b /= factor
    return f"{b:.2f} P{suffix}"


def check_environment() -> None:
    """Menjalankan dan menampilkan laporan diagnostik lingkungan."""
    print("=" * 65)
    print("        DIAGNOSTIK LINGKUNGAN SISTEM & HARDWARE (YOLOv11)")
    print("=" * 65)

    # 1. Informasi Sistem Operasi & Python
    print("\n[1] SISTEM OPERASI & PYTHON")
    print(f"  • Sistem Operasi    : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  • Versi Python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  • Path Executable   : {sys.executable}")

    # 2. Status PyTorch & Komputasi Akselerasi
    print("\n[2] PYTORCH & CUDA ACCELERATION")
    try:
        import torch

        print(f"  • Versi PyTorch     : {torch.__version__}")
        print(f"  • CUDA Terdeteksi   : {'Ya' if torch.cuda.is_available() else 'Tidak (Mode CPU)'}")

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda or "N/A"
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
            print(f"  • Versi CUDA Toolkit: {cuda_version}")
            print(f"  • Versi cuDNN       : {cudnn_version}")
            print(f"  • Jumlah GPU        : {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"    - GPU {i}: {props.name} | VRAM: {vram_gb:.2f} GB | Compute Capability: {props.major}.{props.minor}")
        else:
            print("  • Catatan           : GPU/CUDA tidak terdeteksi di lingkungan lokal ini.")
            print("                        Pelatihan model (train.py) disarankan menggunakan Google Colab GPU.")
    except ImportError:
        print("  • [ERROR] PyTorch belum terpasang.")

    # 3. Pustaka Computer Vision & Machine Learning
    print("\n[3] PUSTAKA COMPUTER VISION & DEPENDENSI")
    packages = [
        ("ultralytics", "Ultralytics YOLOv11"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
    ]

    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Terpasang")
            print(f"  • {label:<20}: {ver}")
        except ImportError:
            print(f"  • {label:<20}: [TIDAK TERPASANG]")

    print("\n" + "=" * 65)
    print("Diagnostik selesai.")
    print("=" * 65)


if __name__ == "__main__":
    check_environment()
