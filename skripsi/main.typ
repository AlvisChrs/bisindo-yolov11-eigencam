// =============================================================================
// STARTER TEMPLATE SKRIPSI - FATISDA UNIVERSITAS SEBELAS MARET
// =============================================================================

// Konfigurasi Halaman & Batas Tepi (Margin) sesuai Pedoman FATISDA UNS:
// Atas 4 cm, Kiri 4 cm, Bawah 3 cm, Kanan 3 cm
#set page(
  paper: "a4",
  margin: (top: 4cm, left: 4cm, bottom: 3cm, right: 3cm),
  header: context {
    let page_num = counter(page).display()
    align(right)[#text(size: 10pt)[#page_num]]
  },
  footer: none,
)

// Konfigurasi Font & Bahasa
#set text(
  font: "Times New Roman",
  size: 12pt,
  lang: "id",
)

// Konfigurasi Paragraf & Alinea:
// Spasi 1.5 (leading 0.75em - 1em tergantung font-size), alinea menjorok 1.5 cm
#set par(
  justify: true,
  leading: 1em,
  first-line-indent: 1.5cm,
)

// Format Judul Bab & Sub-Bab
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(width: 100%, below: 1.5em)[
  #align(center)[
    #text(size: 14pt, weight: "bold")[#upper(it.body)]
  ]
]

#show heading.where(level: 2): it => block(below: 1em)[
  #text(size: 12pt, weight: "bold")[
    #counter(heading).display() #it.body
  ]
]

// =============================================================================
// DOKUMEN UJI COBA INTEGRASI TYPST
// =============================================================================

= BAB 1 PENDAHULUAN

== Latar Belakang Masalah
Ini adalah contoh integrasi Typst di Antigravity IDE. Dokumen ini telah dikonfigurasi mengikuti pedoman resmi tata tulis skripsi Fakultas Teknologi Informasi dan Sains Data (FATISDA), Universitas Sebelas Maret:

1. Ukuran kertas A4 dengan batas tepi atas 4 cm, kiri 4 cm, bawah 3 cm, dan kanan 3 cm.
2. Jenis huruf menggunakan Times New Roman ukuran 12 pt dengan jarak spasi 1,5.
3. Awal alinea menjorok 1,5 cm dari batas tepi kiri dengan perataan teks rata kanan-kiri (*justify*).

== Rumusan Masalah
Berikut adalah contoh penulisan rincian rumusan masalah tanpa menggunakan simbol *bullet*:
1. Bagaimana mengintegrasikan model YOLOv11-nano untuk klasifikasi dan deteksi alfabet BISINDO secara *real-time*?
2. Bagaimana representasi visual *attention map* yang dihasilkan oleh metode Eigen-CAM untuk menjelaskan prediksi model?

== Contoh Persamaan Matematika (XAI / Metrik)
Typst mempermudah penulisan rumus matematika dan persamaan ilmiah:

$ "mAP" = 1/n sum_(k=1)^n "AP"_k $

Dan untuk proyeksi *Singular Value Decomposition* (SVD) pada Eigen-CAM:

$ A = U Sigma V^T $

== Contoh Tabel dan Gambar
Penulisan judul tabel diletakkan di atas tabel, sedangkan judul gambar diletakkan di bawah gambar secara simetris:

#figure(
  table(
    columns: (1fr, 2fr, 2fr),
    align: (center, center, center),
    stroke: 0.5pt,
    table.header([*No*], [*Arsitektur*], [*Target Akurasi (mAP50)*]),
    [1], [YOLOv11-nano], [> 90%],
    [2], [YOLOv11-small], [> 92%],
  ),
  caption: [Perbandingan Rencana Arsitektur Model Pengujian],
) <tab:rencana_model>

Dokumen ini siap digunakan sebagai dasar penulisan skripsi dengan Typst.
