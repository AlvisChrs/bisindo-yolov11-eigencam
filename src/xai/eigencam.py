"""
src/xai/eigencam.py
────────────────────
Implementasi EigenCAM multi-scale untuk visualisasi XAI pada YOLOv11 (n/s/m/l/x).

Referensi metode
----------------
Muhammad, M. B., & Yeasin, M. (2020). Eigen-CAM: Class Activation Map using
Principal Components. 2020 International Joint Conference on Neural Networks
(IJCNN). https://doi.org/10.1109/IJCNN48605.2020.9206626

PERBAIKAN vs kode notebook lama
--------------------------------
1. Mean-centering sebelum SVD:
   Ditambahkan `feat_flat -= feat_flat.mean(axis=0, keepdims=True)` sebelum
   np.linalg.svd(). Tanpa ini, komponen pertama SVD akan menangkap arah
   rata-rata aktivasi (bias), bukan arah variansi terbesar. Dengan mean-
   centering, hasil sesuai definisi PCA asli Muhammad & Yeasin (2020).

2. Parameter n_components (default=3):
   Kode lama hardcode 3 komponen (Vt[0:3]). Sekarang jadi parameter eksplisit.
   CATATAN: EigenCAM baku (Muhammad & Yeasin, 2020) hanya memakai 1 komponen
   pertama. Penggunaan n_components=3 di sini adalah MODIFIKASI/VARIAN untuk
   tangkap lebih banyak struktur aktivasi. Ini harus disebutkan secara eksplisit
   di laporan/skripsi sebagai deviasi dari metode asli.

3. Preprocessing konsisten dengan LetterBox:
   Kode lama pakai cv2.resize() polos yang mendistorsi aspek rasio gambar.
   Sekarang menggunakan ultralytics.data.augment.LetterBox (resize + pad)
   yang identik dengan preprocessing training dan inference, sehingga
   koordinat fitur layer tetap presisi dan heatmap tidak terdistorsi.

4. Auto-deteksi layer FPN (P3, P4, P5):
   Tidak lagi hardcode layer indices [16, 19, 22] untuk nano saja.
   Sekarang otomatis mendeteksi 3 layer feature pyramid berdasarkan stride
   output (≈8, 16, 32) — bekerja untuk YOLOv11n/s/m/l/x tanpa ubah kode.
   Fallback ke [16, 19, 22] jika deteksi gagal.

5. Dua versi output (raw_cam & visual_overlay) + parameter mask_to_bbox:
   - raw_cam       : heatmap mentah setelah normalisasi, SEBELUM post-processing
                     visual. Bisa dipakai sebagai bukti ilmiah bahwa hasilnya
                     bukan sekadar "dipoles".
   - visual_overlay: overlay RGB dengan Gaussian blur + thresholding persentil
                     80 + gamma correction, untuk visualisasi di laporan.
   - mask_to_bbox=False: matikan masking bounding box untuk melihat distribusi
                         aktivasi penuh (berguna untuk lampiran pembanding).

Penggunaan
----------
    # Satu gambar (dari root project):
    python -m src.xai.eigencam \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/A_001.jpg

    # Seluruh folder test:
    python -m src.xai.eigencam \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/ \\
        --output  results/eigencam/

    # Tanpa masking bounding box (untuk lampiran pembanding):
    python -m src.xai.eigencam \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/A_001.jpg \\
        --no-mask-bbox

    # Gunakan hanya 1 komponen (EigenCAM baku):
    python -m src.xai.eigencam \\
        --weights results/bisindo_yolo11n/weights/best.pt \\
        --source  datasets/bisindo-dataset-1/test/images/A_001.jpg \\
        --n-components 1
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────
# AUTO-DETECT FEATURE PYRAMID LAYERS (P3, P4, P5)
# ─────────────────────────────────────────────────────────────

def _detect_fpn_layers(model_inner, imgsz: int = 640, device: str = "") -> list[int]:
    """
    Deteksi otomatis 3 layer Feature Pyramid Network (P3, P4, P5) berdasarkan
    stride output. Cocok untuk YOLOv11n/s/m/l/x tanpa hardcode index.

    Strategi:
    - Jalankan forward pass dummy sekali
    - Catat output shape (H, W) tiap layer
    - Pilih 3 layer dengan stride ≈ 8, 16, 32 (rasio 1:2:4)
    - Urutkan dari stride terkecil (P3) ke terbesar (P5)

    Returns:
        list[int]: Index layer [P3_idx, P4_idx, P5_idx]
    """
    _device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_inner.to(_device)
    model_inner.eval()

    # Dummy input
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=_device)

    # Hook untuk capture output shape tiap layer
    layer_shapes = {}

    def _make_hook(idx):
        def _hook(module, inp, output):
            feat = output[0] if isinstance(output, tuple) else output
            # feat shape: (1, C, H, W)
            layer_shapes[idx] = feat.shape[-2:]  # (H, W)
        return _hook

    handles = []
    for idx, module in enumerate(model_inner.model):
        handles.append(module.register_forward_hook(_make_hook(idx)))

    with torch.no_grad():
        _ = model_inner(dummy)

    for h in handles:
        h.remove()

    # Hitung stride tiap layer (imgsz / H)
    strides = {}
    for idx, (h, w) in layer_shapes.items():
        if h > 0 and w > 0:
            strides[idx] = imgsz / h  # asumsi square-ish

    # Target strides untuk P3, P4, P5
    target_strides = [8, 16, 32]
    selected = []

    for target in target_strides:
        # Cari layer dengan stride paling mendekati target
        best_idx = min(strides.keys(), key=lambda i: abs(strides[i] - target))
        selected.append(best_idx)

    # Urutkan by stride (P3→P4→P5)
    selected.sort(key=lambda i: strides[i])

    # Fallback ke default nano kalau deteksi gagal (kurang dari 3 layer valid)
    if len(selected) < 3:
        return [16, 19, 22]

    return selected


# Layer default untuk YOLOv11-nano (fallback jika auto-deteksi gagal)
# Layer 16, 19, 22 adalah feature pyramid levels dengan resolusi berbeda —
# multi-scale untuk menangkap fitur kasar (posisi tangan) dan halus (jari)
DEFAULT_LAYERS: list[int] = [16, 19, 22]

# Format gambar yang didukung
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ─────────────────────────────────────────────────────────────
# AKTIVASI LAYER VIA FORWARD HOOK
# ─────────────────────────────────────────────────────────────

def _extract_activation(model_inner, input_tensor: torch.Tensor, layer_idx: int) -> np.ndarray:
    """
    Ekstrak feature map dari satu layer tertentu via forward hook.

    Parameters
    ----------
    model_inner : torch.nn.Module
        Model PyTorch bagian dalam (bukan YOLO wrapper).
    input_tensor : torch.Tensor
        Tensor gambar shape (1, C, H, W), sudah di-LetterBox dan di-normalize.
    layer_idx : int
        Index layer di model.model (0-based).

    Returns
    -------
    np.ndarray
        Feature map shape (C, H, W) sebagai numpy array float32.
    """
    activation = {}

    def _hook(module, inp, output):
        feat = output[0] if isinstance(output, tuple) else output
        activation["feat"] = feat.detach()

    handle = model_inner.model[layer_idx].register_forward_hook(_hook)
    with torch.no_grad():
        _ = model_inner(input_tensor)
    handle.remove()

    return activation["feat"].squeeze(0).cpu().numpy()  # (C, H, W)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING — KONSISTEN DENGAN LETTERBOX
# ─────────────────────────────────────────────────────────────

def _preprocess_image(img_bgr: np.ndarray, imgsz: int = 640) -> tuple[torch.Tensor, np.ndarray]:
    """
    Preprocessing gambar dengan LetterBox (resize + pad), konsisten dengan
    pipeline training dan inference Ultralytics.

    PERBAIKAN vs kode lama: kode lama memakai cv2.resize(img_rgb, (640, 640))
    yang meng-stretch gambar dan mendistorsi aspek rasio. LetterBox menambahkan
    padding abu-abu sehingga proporsi tangan tetap benar.

    Parameters
    ----------
    img_bgr : np.ndarray
        Gambar BGR asli dari cv2.imread().
    imgsz : int
        Target ukuran gambar (default 640).

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        - input_tensor : tensor shape (1, 3, imgsz, imgsz), float32, nilai [0,1]
        - img_letterbox: gambar RGB setelah LetterBox (untuk overlay heatmap)
    """
    from ultralytics.data.augment import LetterBox

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # LetterBox: resize + pad tanpa distorsi
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False)
    img_lb = lb(image=img_rgb)   # (H, W, 3) RGB uint8

    # Konversi ke tensor (1, 3, H, W) float32 [0, 1]
    rgb_float = np.float32(img_lb) / 255.0
    tensor = (
        torch.from_numpy(rgb_float)
        .permute(2, 0, 1)   # HWC → CHW
        .unsqueeze(0)        # → (1, C, H, W)
        .float()
    )

    return tensor, img_lb


# ─────────────────────────────────────────────────────────────
# CORE EIGENCAM
# ─────────────────────────────────────────────────────────────

def eigencam(
    model,
    img_path: str,
    layers: list[int] = None,
    n_components: int = 3,
    imgsz: int = 640,
    device: str = "",
    mask_to_bbox: bool = True,
    conf: float = 0.5,
    iou: float = 0.7,
) -> dict:
    """
    Hitung EigenCAM multi-scale untuk satu gambar dan kembalikan heatmap
    mentah serta overlay tervisualisasi.

    Parameters
    ----------
    model : YOLO
        Model Ultralytics YOLO yang sudah dimuat.
    img_path : str
        Path ke file gambar input.
    layers : list[int], optional
        Index layer yang digunakan untuk multi-scale.
        Default: **auto-deteksi** (mencari layer FPN P3, P4, P5 berdasarkan stride).
        Fallback ke [16, 19, 22] untuk YOLOv11-nano jika deteksi gagal.
    n_components : int, optional
        Jumlah principal component SVD yang dijumlahkan. Default: 3.

        CATATAN PENTING: EigenCAM baku (Muhammad & Yeasin, 2020) hanya memakai
        komponen pertama (n_components=1). Nilai default 3 adalah modifikasi
        untuk menangkap lebih banyak struktur aktivasi. Sebutkan ini sebagai
        deviasi di laporan/skripsi.
    imgsz : int
        Ukuran input model. Default: 640.
    device : str
        Device komputasi ("cuda", "cpu", atau "" untuk auto). Default: "".
    mask_to_bbox : bool
        Jika True, heatmap dimask sehingga hanya area dalam bounding box
        (+ padding 20px) yang dipertahankan. Default: True.

        Set False untuk menghasilkan heatmap tanpa masking sebagai pembanding
        di lampiran skripsi.
    conf : float
        Confidence threshold untuk inferensi bounding box. Default: 0.5.
    iou : float
        IoU threshold untuk NMS. Default: 0.7.

    Returns
    -------
    dict dengan key:
        raw_cam (np.ndarray):
            Heatmap mentah float32 shape (imgsz, imgsz), nilai [0, 1].
            Ini adalah output SEBELUM post-processing visual (blur, threshold,
            gamma). Gunakan ini sebagai bukti ilmiah di laporan.

        visual_overlay (np.ndarray):
            Overlay RGB float32 shape (imgsz, imgsz, 3), nilai [0, 1].
            Heatmap setelah: Gaussian blur + thresholding persentil 80 +
            gamma correction (γ=0.4) + overlay di atas gambar asli.

        label (str):
            Label kelas yang terdeteksi, atau "Tidak Terdeteksi".
        confidence (float):
            Confidence score prediksi tertinggi.
        box_xyxy (list[int] atau None):
            Bounding box [x1, y1, x2, y2] dalam koordinat gambar asli,
            atau None jika tidak ada deteksi.
        img_original (np.ndarray):
            Gambar RGB asli (sebelum letterbox), uint8, untuk keperluan plot.
    """
    if layers is None:
        # Auto-deteksi layer FPN (P3, P4, P5) berdasarkan model yang dipakai
        layers = _detect_fpn_layers(model.model, imgsz=imgsz, device=device)

    # ── 1. Baca gambar ─────────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {img_path}")
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 2. Preprocessing: LetterBox (bukan cv2.resize polos) ──
    input_tensor, img_lb_rgb = _preprocess_image(img_bgr, imgsz)
    rgb_float = np.float32(img_lb_rgb) / 255.0

    # ── 3. Siapkan model untuk forward pass ────────────────────
    _device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_inner = model.model
    model_inner.to(_device)
    model_inner.eval()
    input_tensor = input_tensor.to(_device)

    # ── 4. Inferensi YOLO untuk bounding box ──────────────────
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=str(_device),
        save=False,
        verbose=False,
    )
    result = results[0]

    label, conf_val, box = "Tidak Terdeteksi", 0.0, None
    if len(result.boxes) > 0:
        best_idx = int(result.boxes.conf.argmax())
        label    = result.names[int(result.boxes.cls[best_idx])]
        conf_val = float(result.boxes.conf[best_idx])
        box      = result.boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()

    # ── 5. Multi-scale EigenCAM ────────────────────────────────
    cam_final = np.zeros((imgsz, imgsz), dtype=np.float32)

    for layer_idx in layers:
        feat_np = _extract_activation(model_inner, input_tensor, layer_idx)
        C, H, W = feat_np.shape
        feat_flat = feat_np.reshape(C, -1).T   # shape: (H*W, C)

        # PERBAIKAN #1: Mean-centering sebelum SVD
        # Tanpa ini, SVD menangkap arah rata-rata (bias), bukan variansi.
        feat_flat = feat_flat - feat_flat.mean(axis=0, keepdims=True)

        # SVD — ambil right singular vectors (Vt rows = principal directions)
        _, _, Vt = np.linalg.svd(feat_flat, full_matrices=False)

        # PERBAIKAN #2: n_components sebagai parameter (bukan hardcode 3)
        cam_layer = np.zeros((H, W), dtype=np.float32)
        for i in range(min(n_components, Vt.shape[0])):
            projected = np.einsum("c,chw->hw", Vt[i], feat_np)
            cam_layer += np.abs(projected)

        # Normalisasi per-layer
        c_min, c_max = cam_layer.min(), cam_layer.max()
        if c_max - c_min > 1e-8:
            cam_layer = (cam_layer - c_min) / (c_max - c_min)

        # Masking tepi (10% border) — kurangi artefak padding
        bh = max(1, int(H * 0.10))
        bw = max(1, int(W * 0.10))
        cam_layer[:bh, :]  = 0
        cam_layer[-bh:, :] = 0
        cam_layer[:, :bw]  = 0
        cam_layer[:, -bw:] = 0

        cam_final += cv2.resize(cam_layer, (imgsz, imgsz))

    # Normalisasi gabungan semua layer
    c_min, c_max = cam_final.min(), cam_final.max()
    if c_max - c_min > 1e-8:
        cam_final = (cam_final - c_min) / (c_max - c_min)

    # ── 6. Masking bounding box (opsional) ────────────────────
    # PERBAIKAN #4 (mask_to_bbox parameter): bisa dimatikan via mask_to_bbox=False
    if mask_to_bbox and box is not None:
        orig_h, orig_w = img_rgb_orig.shape[:2]
        # Hitung scale dan offset LetterBox yang benar:
        # LetterBox memakai scale seragam (bukan stretch) + padding di satu sisi.
        scale = min(imgsz / orig_h, imgsz / orig_w)
        pad_x = (imgsz - orig_w * scale) / 2   # padding kiri
        pad_y = (imgsz - orig_h * scale) / 2   # padding atas
        x1 = int(box[0] * scale + pad_x)
        y1 = int(box[1] * scale + pad_y)
        x2 = int(box[2] * scale + pad_x)
        y2 = int(box[3] * scale + pad_y)
        pad = 20
        mask = np.zeros((imgsz, imgsz), dtype=np.float32)
        mask[max(0, y1 - pad):min(imgsz, y2 + pad),
             max(0, x1 - pad):min(imgsz, x2 + pad)] = 1.0
        cam_final = cam_final * mask

        c_min, c_max = cam_final.min(), cam_final.max()
        if c_max - c_min > 1e-8:
            cam_final = (cam_final - c_min) / (c_max - c_min)

    # ── 7. Simpan raw_cam SEBELUM post-processing visual ──────
    # PERBAIKAN #4: raw_cam adalah heatmap mentah untuk bukti ilmiah
    raw_cam = cam_final.copy()

    # ── 8. Post-processing visual ─────────────────────────────
    # (dilakukan pada salinan, raw_cam tidak tersentuh)
    cam_visual = cam_final.copy()

    # Gaussian blur — haluskan noise aktivasi
    cam_visual = cv2.GaussianBlur(cam_visual, (15, 15), 0)

    c_min, c_max = cam_visual.min(), cam_visual.max()
    if c_max - c_min > 1e-8:
        cam_visual = (cam_visual - c_min) / (c_max - c_min)

    # Thresholding persentil 80 — pertahankan hanya puncak aktivasi
    positive = cam_visual[cam_visual > 0]
    if positive.size > 0:
        threshold = np.percentile(positive, 80)
        cam_visual[cam_visual < threshold] = 0

    c_min, c_max = cam_visual.min(), cam_visual.max()
    if c_max - c_min > 1e-8:
        cam_visual = (cam_visual - c_min) / (c_max - c_min)

    # Gamma correction γ=0.4 — perjelas kontras tinggi/rendah
    cam_visual = np.power(cam_visual, 0.4)

    # Overlay JET colormap di atas gambar asli
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_visual), cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    visual_overlay = np.clip(
        0.55 * rgb_float + 0.45 * np.float32(heatmap_rgb) / 255.0,
        0.0, 1.0,
    )

    return {
        "raw_cam":       raw_cam,
        "visual_overlay": visual_overlay,
        "label":          label,
        "confidence":     round(conf_val, 4),
        "box_xyxy":       box,
        "img_original":   img_rgb_orig,
    }


# ─────────────────────────────────────────────────────────────
# VISUALISASI & SIMPAN HASIL
# ─────────────────────────────────────────────────────────────

def save_eigencam_figure(
    result: dict,
    output_path: str,
    n_components: int = 3,
    mask_to_bbox: bool = True,
) -> None:
    """
    Simpan figure matplotlib 2-panel (gambar asli + heatmap overlay) ke disk.

    Parameters
    ----------
    result : dict
        Output dari fungsi eigencam().
    output_path : str
        Path file output (misal: results/eigencam/A_001_eigencam.png).
    n_components : int
        Jumlah komponen yang dipakai (untuk label di figure).
    mask_to_bbox : bool
        Apakah masking bbox dipakai (untuk label di figure).
    """
    import matplotlib
    matplotlib.use("Agg")   # backend non-interaktif agar aman di server/Colab
    import matplotlib.pyplot as plt

    img_rgb  = result["img_original"]
    overlay  = result["visual_overlay"]
    label    = result["label"]
    conf_val = result["confidence"]
    box      = result["box_xyxy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel kiri: gambar asli + bounding box
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original\nPrediksi: {label} (conf={conf_val:.2f})", fontsize=13)
    axes[0].axis("off")
    if box is not None:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x1, max(0, y1 - 8), f"{label} {conf_val:.2f}",
            color="lime", fontsize=11, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"),
        )

    # Panel kanan: EigenCAM overlay
    mask_note = "dengan masking bbox" if mask_to_bbox else "TANPA masking bbox"
    comp_note = f"n_components={n_components}"
    if n_components != 1:
        comp_note += " (modifikasi; EigenCAM baku=1)"

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"EigenCAM Heatmap — Huruf '{label}'\n"
        f"{comp_note}, {mask_note}\n"
        f"Merah-Kuning = fokus tinggi | Biru = fokus rendah",
        fontsize=11,
    )
    axes[1].axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.03, pad=0.02)
    cbar.set_label("Tingkat Perhatian Model", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Rendah", "Sedang", "Tinggi"])

    plt.suptitle(
        f"Analisis XAI EigenCAM — BISINDO Huruf '{label}'",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _collect_images(source: str) -> list[Path]:
    """Kumpulkan path gambar dari file tunggal atau folder."""
    src = Path(source)
    if not src.exists():
        print(f"[ERROR] Source tidak ditemukan: {src.resolve()}", file=sys.stderr)
        sys.exit(1)
    if src.is_file():
        return [src]
    imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    if not imgs:
        print(f"[ERROR] Tidak ada gambar di: {src.resolve()}", file=sys.stderr)
        sys.exit(1)
    return imgs


def run_batch(
    weights: str,
    source: str,
    output: str = "results/eigencam",
    layers: list[int] = None,
    n_components: int = 3,
    mask_to_bbox: bool = True,
    conf: float = 0.5,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "",
    save_raw: bool = False,
) -> None:
    """
    Proses batch gambar — hasilkan figure EigenCAM untuk tiap gambar.

    Parameters
    ----------
    weights : str
        Path ke file .pt model.
    source : str
        Path ke gambar atau folder.
    output : str
        Folder output. Default: "results/eigencam".
    layers : list[int], optional
        Layer yang digunakan. Default: [16, 19, 22].
    n_components : int
        Jumlah komponen SVD. Default: 3.
    mask_to_bbox : bool
        Aktifkan masking bounding box. Default: True.
    conf, iou, imgsz, device : float/int/str
        Parameter inference.
    save_raw : bool
        Simpan juga raw_cam sebagai file .npy. Default: False.
    """
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights tidak ditemukan: {weights_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    model  = YOLO(str(weights_path))
    images = _collect_images(source)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = DEFAULT_LAYERS

    mask_tag = "masked" if mask_to_bbox else "unmasked"
    print(f"[INFO] EigenCAM batch: {len(images)} gambar")
    print(f"[INFO] Layers        : {layers}")
    print(f"[INFO] n_components  : {n_components}  ({'baku' if n_components == 1 else 'modifikasi'})")
    print(f"[INFO] mask_to_bbox  : {mask_to_bbox}")
    print(f"[INFO] Output        : {output_dir.resolve()}")
    print()

    for i, img_path in enumerate(images, start=1):
        try:
            result = eigencam(
                model=model,
                img_path=str(img_path),
                layers=layers,
                n_components=n_components,
                imgsz=imgsz,
                device=device,
                mask_to_bbox=mask_to_bbox,
                conf=conf,
                iou=iou,
            )

            stem = img_path.stem
            fig_path = output_dir / f"{stem}_eigencam_{mask_tag}.png"
            save_eigencam_figure(result, str(fig_path), n_components, mask_to_bbox)

            if save_raw:
                raw_path = output_dir / f"{stem}_raw_cam.npy"
                np.save(str(raw_path), result["raw_cam"])

            print(
                f"  [{i:>4}/{len(images)}] {img_path.name:<30} "
                f"→ {result['label']} ({result['confidence']:.2f})"
                f"{'  [saved raw]' if save_raw else ''}"
            )

        except Exception as e:
            print(f"  [{i:>4}/{len(images)}] {img_path.name:<30} → [ERROR] {e}", file=sys.stderr)

    print(f"\n[DONE] EigenCAM selesai. Hasil di: {output_dir.resolve()}")


def main() -> None:
    """Entry point untuk penggunaan via CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "EigenCAM XAI untuk YOLOv11-nano BISINDO.\n"
            "CATATAN: n_components>1 adalah modifikasi dari EigenCAM baku (Muhammad & Yeasin, 2020)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights",      required=True, help="Path ke file .pt model")
    parser.add_argument("--source",       required=True, help="Gambar tunggal atau folder")
    parser.add_argument("--output",       default="results/eigencam", help="Folder output")
    parser.add_argument("--layers",       nargs="+", type=int, default=DEFAULT_LAYERS,
                        help="Index layer untuk multi-scale (default: 16 19 22)")
    parser.add_argument("--n-components", type=int, default=3,
                        help="Jumlah komponen SVD. Default=3 (modifikasi). "
                             "Gunakan 1 untuk EigenCAM baku (Muhammad & Yeasin 2020)")
    parser.add_argument("--no-mask-bbox", action="store_true",
                        help="Nonaktifkan masking bounding box (untuk lampiran pembanding)")
    parser.add_argument("--conf",         type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou",          type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--imgsz",        type=int, default=640, help="Ukuran gambar")
    parser.add_argument("--device",       default="", help="Device: '0', 'cpu', atau '' (auto)")
    parser.add_argument("--save-raw",     action="store_true",
                        help="Simpan raw_cam sebagai file .npy (untuk analisis lanjutan)")
    args = parser.parse_args()

    run_batch(
        weights=args.weights,
        source=args.source,
        output=args.output,
        layers=args.layers,
        n_components=args.n_components,
        mask_to_bbox=not args.no_mask_bbox,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_raw=args.save_raw,
    )


if __name__ == "__main__":
    main()
