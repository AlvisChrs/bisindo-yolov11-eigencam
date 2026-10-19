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
