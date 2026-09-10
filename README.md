# Deteksi Alfabet BISINDO dengan YOLOv11-nano + EigenCAM XAI

Repositori ini berisi kode untuk skripsi:

> **"Deteksi Alfabet Bahasa Isyarat Indonesia (BISINDO) Menggunakan YOLOv11-nano dengan Pendekatan Explainable Artificial Intelligence (XAI) Berbasis EigenCAM"**

Model mendeteksi 26 huruf alfabet A–Z dalam Bahasa Isyarat Indonesia (BISINDO) secara real-time, dilengkapi analisis transparansi menggunakan EigenCAM untuk memvisualisasikan bagian gambar yang menjadi fokus perhatian model.

---

## Struktur Repo

```
bisindo-yolov11-eigencam/
├── README.md
├── .gitignore
├── .env.example               # template API key (aman di-commit)
├── requirements.txt           # dependensi dengan pin versi
├── configs/
│   └── train_config.yaml      # semua hyperparameter training
├── src/
│   ├── data/
│   │   ├── download_dataset.py        # unduh dataset dari Roboflow
│   │   └── visualize_augmentation.py  # preview grid augmentasi (laporan skripsi)
│   ├── training/
│   │   └── train.py              # training YOLOv11-nano (satu run utuh)
│   ├── evaluation/
│   │   └── evaluate.py           # hitung mAP, P, R, confusion matrix
│   ├── inference/
│   │   └── predict.py            # deteksi satu gambar atau folder
│   └── xai/
│       └── eigencam.py           # visualisasi EigenCAM
├── legacy/
│   └── BISINDO_YOLOv11.ipynb  # notebook Colab asli (arsip referensi)
├── notebooks/
│   └── demo.ipynb             # (opsional) demo visual interaktif
└── results/                   # output eksperimen (gitignored)
```

---

## Instalasi

### 1. Clone repo dan masuk ke folder

```bash
git clone https://github.com/<username>/bisindo-yolov11-eigencam.git
cd bisindo-yolov11-eigencam
```

### 2. Buat virtual environment (opsional tapi dianjurkan)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependensi

**Lokal (CPU — untuk development dan sanity check):**
```bash
pip install -r requirements.txt
```

**Google Colab / NVIDIA GPU (untuk training penuh 40 epoch):**
```bash
pip install -r requirements.txt
# Override torch dengan versi CUDA:
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

### 4. Siapkan API key Roboflow

```bash
cp .env.example .env
# Buka .env dan isi ROBOFLOW_API_KEY dengan key dari https://app.roboflow.com
```

---

## Cara Pakai

### Download Dataset

```bash
python -m src.data.download_dataset
# Dataset tersimpan di: datasets/bisindo-dataset-1/
```

### Visualisasi Augmentasi

Membuat grid 2×3 (5 panel) yang menampilkan teknik augmentasi data yang digunakan saat training,
lengkap dengan parameter yang sama persis seperti `configs/train_config.yaml`.
Hasil disimpan di `results/figures/augmentation_preview.png` (DPI 220, siap laporan skripsi).

```bash
# Default: gambar pertama di train/images/, seed 42
python -m src.data.visualize_augmentation

# Pilih gambar sampel lain (indeks ke-100) dan seed berbeda
python -m src.data.visualize_augmentation --sample-idx 100 --seed 7
```

**Panel yang dihasilkan:**

| Panel | Teknik | Parameter |
|-------|--------|-----------|
| 1 | Original + Bounding Box | — |
| 2 | Horizontal Flip | `fliplr = 0.5` |
| 3 | HSV Augmentation | `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` |
| 4 | Mosaic (4 gambar) | `mosaic = 1.0` |
| 5 | Random Erasing | `erasing = 0.4` |

### Training

```bash
# Training 40 epoch (sesuai configs/train_config.yaml):
python -m src.training.train

# Sanity check 1 epoch:
python -m src.training.train --epochs 1 --name bisindo_sanity

# Resume jika runtime terputus:
python -m src.training.train --resume
```

### Evaluasi

```bash
python -m src.evaluation.evaluate \
    --weights results/bisindo_yolo11n/weights/best.pt

# Output tersimpan di results/evaluation/:
# - metrics.json
# - metrics.txt  (ringkasan siap dikutip di laporan)
# - confusion_matrix.png
```

### Inference

```bash
# Satu gambar:
python -m src.inference.predict \
    --weights results/bisindo_yolo11n/weights/best.pt \
    --source  datasets/bisindo-dataset-1/test/images/A_001.jpg

# Seluruh folder test:
python -m src.inference.predict \
    --weights results/bisindo_yolo11n/weights/best.pt \
    --source  datasets/bisindo-dataset-1/test/images/ \
    --save-json
```

### EigenCAM XAI

```bash
# Visualisasi standar (dengan masking bounding box):
python -m src.xai.eigencam \
    --weights results/bisindo_yolo11n/weights/best.pt \
    --source  datasets/bisindo-dataset-1/test/images/A_001.jpg \
    --output  results/eigencam/

# Tanpa masking bbox (untuk lampiran pembanding di skripsi):
python -m src.xai.eigencam \
    --weights results/bisindo_yolo11n/weights/best.pt \
    --source  datasets/bisindo-dataset-1/test/images/ \
    --no-mask-bbox

# EigenCAM baku 1 komponen (Muhammad & Yeasin, 2020):
python -m src.xai.eigencam \
    --weights results/bisindo_yolo11n/weights/best.pt \
    --source  datasets/bisindo-dataset-1/test/images/A_001.jpg \
    --n-components 1

# Simpan raw heatmap sebagai .npy (untuk analisis kuantitatif):
python -m src.xai.eigencam \
    --weights results/bisindo_yolo11n/weights/best.pt \
    --source  datasets/bisindo-dataset-1/test/images/ \
    --save-raw
```

---

## Catatan Teknis

### Perbaikan vs Kode Lama (Notebook Colab)

| Aspek | Kode Lama | Repo Ini |
|---|---|---|
| API key | Hardcoded di kode | Dibaca dari `.env` via `python-dotenv` |
| Training | Dua sesi (27+13 epoch), LR scheduler restart | Satu run utuh 40 epoch |
| Resume | `YOLO("last.pt").train(epochs=13)` → run baru | `model.train(resume=True)` resmi |
| Preprocessing | `cv2.resize()` polos, distorsi aspek rasio | `LetterBox` (resize+pad, konsisten dengan training) |
| EigenCAM SVD | Tanpa mean-centering | Mean-centering sebelum SVD (sesuai definisi PCA) |

### EigenCAM: Modifikasi vs Implementasi Baku

Implementasi di repo ini menggunakan `n_components=3` sebagai default (menjumlahkan 3 principal component pertama dari SVD). Ini adalah **modifikasi** dari:

> Muhammad, M. B., & Yeasin, M. (2020). Eigen-CAM: Class Activation Map using Principal Components. *2020 International Joint Conference on Neural Networks (IJCNN)*. https://doi.org/10.1109/IJCNN48605.2020.9206626

EigenCAM baku hanya menggunakan **komponen pertama** (`n_components=1`). Penggunaan 3 komponen bertujuan menangkap lebih banyak struktur aktivasi pada fitur multi-scale YOLOv11. Gunakan `--n-components 1` untuk implementasi yang sesuai paper asli.

---

## Hyperparameter Training

Semua hyperparameter ada di `configs/train_config.yaml`:

| Parameter | Nilai |
|---|---|
| Model | YOLOv11-nano (`yolo11n.pt`) |
| Epochs | 40 |
| Batch size | 16 |
| Image size | 640×640 |
| Optimizer | AdamW |
| Learning rate (lr0) | 0.000333 |
| Confidence threshold | 0.5 |
| IoU threshold | 0.7 |

---

## Bobot Model

File `.pt` tidak disimpan di repositori ini karena ukurannya besar.
Bobot hasil training tersedia di: *(isi dengan link Google Drive atau GitHub Releases setelah training selesai)*

---

## Dataset

Dataset BISINDO A-Z diunduh dari Roboflow Universe:

**Citasi (BibTeX):**
```bibtex
@misc{bisindo-dataset_dataset,
  title        = {bisindo-dataset Dataset},
  author       = {bisindo},
  year         = {2024},
  url          = {https://universe.roboflow.com/bisindo/bisindo-dataset},
  note         = {Roboflow Universe. Retrieved via Roboflow API.}
}
```

Lisensi dataset: **CC BY 4.0** — bebas digunakan dengan atribusi.
Lisensi kode di repositori ini: **MIT** — lihat file `LICENSE`.

---

## Referensi

- Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR 2016*.
- Muhammad, M. B., & Yeasin, M. (2020). Eigen-CAM: Class Activation Map using Principal Components. *IJCNN 2020*. https://doi.org/10.1109/IJCNN48605.2020.9206626
- Ultralytics. (2024). YOLO11. https://github.com/ultralytics/ultralytics

---

## Kontributor

| Nama | NIM | Peran |
|------|-----|-------|
| Alvis Marcell Christian | L0123016 | Peneliti utama / pengembang |
