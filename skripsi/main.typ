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

#include "chapters/bab1.typ"
#include "chapters/bab2.typ"
#include "chapters/bab3.typ"
#include "chapters/bab4.typ"
#include "chapters/bab5.typ"

#bibliography("pustaka.bib", title: "DAFTAR PUSTAKA", style: "ieee")


