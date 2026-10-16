#import "template.typ": fatisda-skripsi

#show: fatisda-skripsi.with(
  title: "Deteksi Alfabet Bahasa Isyarat Indonesia (BISINDO) Menggunakan YOLOv11-Nano Terintegrasi EigenCAM sebagai Explainable Artificial Intelligence (XAI)",
  author: "Alvis Marcell Christian",
  nim: "L0123016",
  prodi: "S1 Informatika",
  fakultas: "Fakultas Teknologi Informasi dan Sains Data",
  universitas: "Universitas Sebelas Maret",
  kota: "Surakarta",
  tahun: "2026",
  bulan_tahun: "Oktober 2026",
  tipe_dokumen: "proposal", // ganti ke "skripsi" saat sidang skripsi
  
  // Dosen Pembimbing & Penguji (silakan sesuaikan nama dan NIP dosen Anda)
  pembimbing1: (nama: "Nama Dosen Pembimbing I, Gelar", nip: "19700101XXXXXXXXXX"),
  pembimbing2: (nama: "Nama Dosen Pembimbing II, Gelar", nip: "19750101XXXXXXXXXX"),
  penguji_utama: (nama: "Nama Dosen Penguji Utama, Gelar", nip: "19800101XXXXXXXXXX"),
  koordinator_skripsi: (nama: "Nama Koordinator Skripsi, Gelar", nip: "19720101XXXXXXXXXX"),
  kaprodi: (nama: "Nama Kaprodi Informatika, Gelar", nip: "19680101XXXXXXXXXX"),
  
  tampilkan_daftar_isi: true,
  tampilkan_daftar_gambar: true,
  tampilkan_daftar_tabel: true,
)

// =============================================================================
// NASKAH UTAMA
// =============================================================================

#set heading(numbering: "1.1")

= BAB 1 PENDAHULUAN

== Latar Belakang Masalah
Bahasa Isyarat Indonesia (BISINDO) merupakan salah satu media komunikasi utama bagi komunitas Tuli di Indonesia. Sistem penerjemahan otomatis berbasis *computer vision* dapat menjembatani komunikasi antara teman Tuli dan masyarakat umum.

Dokumen ini disusun menggunakan template Typst FATISDA UNS yang telah mengunci seluruh aturan pedoman penulisan:
1. Batas tepi pengetikan (margin): atas 4 cm, kiri 4 cm, bawah 3 cm, dan kanan 3 cm.
2. Huruf menggunakan jenis Times New Roman ukuran 12 pt dengan jarak spasi 1,5.
3. Penomoran halaman angka Romawi kecil untuk halaman awal (tengah bawah) dan angka Arab untuk naskah utama (kanan atas, kecuali awal bab di tengah bawah).
4. Tulisan program studi, usulan penelitian, lambang UNS, dan identitas fakultas tercetak lengkap pada cover sesuai revisi dosen.

== Rumusan Masalah
Berdasarkan latar belakang tersebut, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana mengimplementasikan arsitektur YOLOv11-nano untuk mendeteksi alfabet BISINDO secara akurat dan *real-time*?
2. Bagaimana visualisasi atensi model menggunakan metode Eigen-CAM sebagai pendekatan *Explainable AI* (XAI)?

== Batasan Masalah
Batasan masalah pada penelitian ini mencakup:
1. Dataset mencakup alfabet statis BISINDO (A hingga Z).
2. Model yang digunakan dibatasi pada varian YOLOv11-nano.

== Tujuan Penelitian
Tujuan yang ingin dicapai adalah:
1. Mengembangkan model deteksi gestur alfabet BISINDO berbasis YOLOv11-nano.
2. Menganalisis representasi visual fitur menggunakan Eigen-CAM.

== Manfaat Penelitian
Manfaat dari penelitian ini diharapkan dapat memberikan kontribusi pada pengembangan teknologi aksesibilitas untuk komunitas Tuli serta memberikan transparansi model *deep learning* melalui XAI.

== Sistematika Penulisan
Sistematika penulisan proposal skripsi ini terbagi menjadi:
1. *BAB 1 PENDAHULUAN:* Memuat latar belakang, rumusan masalah, batasan, tujuan, manfaat, dan sistematika penulisan.
2. *BAB 2 TINJAUAN PUSTAKA:* Memuat landasan teori mengenai BISINDO, YOLOv11, Eigen-CAM, dan penelitian terkait.
3. *BAB 3 METODOLOGI PENELITIAN:* Memuat tahapan pengumpulan dataset, pra-pemrosesan, pelatihan, dan pengujian model.

= BAB 2 TINJAUAN PUSTAKA

== Landasan Teori
Bagian ini menguraikan konsep-konsep dasar yang melandasi penelitian.

=== Bahasa Isyarat Indonesia (BISINDO)
BISINDO adalah bahasa isyarat alami yang berkembang di kalangan komunitas Tuli Indonesia.

=== Arsitektur YOLOv11-nano
YOLOv11 memperkenalkan peningkatan efisiensi ekstraksi fitur melalui modul C3k2 dan SPPF.

=== Explainable Artificial Intelligence (Eigen-CAM)
Eigen-CAM memanfaatkan *Principal Component Analysis* (PCA) atau *Singular Value Decomposition* (SVD) pada matriks aktivasi konvolusi:

$ A = U Sigma V^T $

== Penelitian Terkait
Berikut adalah tabel ringkasan perbandingan penelitian terdahulu:

#figure(
  table(
    columns: (0.8fr, 2fr, 2fr, 2fr),
    align: (center, left, left, center),
    stroke: 0.5pt,
    table.header([*No*], [*Peneliti*], [*Metode*], [*Akurasi*]),
    [1], [Peneliti A (2023)], [YOLOv8], [91.2% mAP50],
    [2], [Peneliti B (2024)], [MediaPipe + SVM], [88.5%],
    [3], [Penelitian Ini], [YOLOv11-nano + Eigen-CAM], [Target > 92%],
  ),
  caption: [Perbandingan Penelitian Terkait Deteksi Bahasa Isyarat],
) <tab:penelitian_terkait>

= BAB 3 METODOLOGI PENELITIAN

== Pengumpulan dan Persiapan Dataset
Dataset dikumpulkan melalui rekaman gestur alfabet BISINDO dari berbagai sudut pandang dan kondisi pencahayaan.

== Evaluasi Kinerja Model
Metrik evaluasi yang digunakan meliputi *Mean Average Precision* (mAP):

$ "mAP" = 1/n sum_(k=1)^n "AP"_k $
