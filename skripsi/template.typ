// =============================================================================
// TEMPLATE SKRIPSI & PROPOSAL SKRIPSI
// FAKULTAS TEKNOLOGI INFORMASI DAN SAINS DATA (FATISDA)
// UNIVERSITAS SEBELAS MARET (UNS) SURAKARTA
// Berdasarkan Pedoman Resmi Penulisan Skripsi FATISDA UNS (Juli 2025)
// =============================================================================

#let fatisda-skripsi(
  // Metadata Utama
  title: "",
  author: "",
  nim: "",
  prodi: "S1 Informatika",
  fakultas: "Fakultas Teknologi Informasi dan Sains Data",
  universitas: "Universitas Sebelas Maret",
  kota: "Surakarta",
  tahun: "2026",
  bulan_tahun: "Oktober 2026",
  
  // Tipe Dokumen: "proposal" (untuk Sempro) atau "skripsi" (untuk Semhas / Sidang)
  tipe_dokumen: "proposal",
  
  // Dosen Pembimbing & Penguji
  pembimbing1: (nama: "", nip: ""),
  pembimbing2: (nama: "", nip: ""),
  penguji_utama: (nama: "", nip: ""),
  penguji_pendamping: (nama: "", nip: ""),
  koordinator_skripsi: (nama: "", nip: ""),
  kaprodi: (nama: "", nip: ""),
  tanggal_persetujuan: ".................... 2026",
  tanggal_seminar: ".................... 2026",

  // Halaman Pelengkap Awal (Opsional)
  motto: none,
  persembahan: none,
  abstrak_id: none,
  abstrak_en: none,
  kata_pengantar: none,
  
  // Daftar Isi, Gambar, Tabel
  tampilkan_daftar_isi: true,
  tampilkan_daftar_gambar: true,
  tampilkan_daftar_tabel: true,
  tampilkan_daftar_lampiran: false,
  
  // Konten Naskah Utama
  body
) = {
  // ---------------------------------------------------------------------------
  // 1. PENGATURAN UMUM DOKUMEN & TIPOGRAFI
  // ---------------------------------------------------------------------------
  set document(title: title, author: author)
  
  set text(
    font: "Times New Roman",
    size: 12pt,
    lang: "id",
  )

  // ---------------------------------------------------------------------------
  // 2. ATURAN GAMBAR, TABEL, DAN PERSAMAAN
  // ---------------------------------------------------------------------------
  // Aturan Pedoman Hal. 40 & 61:
  // - Tabel: judul di ATAS tabel, 12pt reguler (tidak dibold), center, tanpa titik.
  // - Gambar: judul di BAWAH gambar, 12pt reguler (tidak dibold), center, tanpa titik.
  show figure.where(kind: table): it => {
    set align(center)
    block(spacing: 1.5em)[
      #if it.has("caption") [
        #block(below: 0.8em)[
          #text(size: 12pt, weight: "regular")[
            #it.supplement #it.counter.display(it.numbering) #it.caption.body
          ]
        ]
      ]
      #it.body
    ]
  }

  show figure.where(kind: image): it => {
    set align(center)
    block(spacing: 1.5em)[
      #it.body
      #if it.has("caption") [
        #block(above: 0.8em)[
          #text(size: 12pt, weight: "regular")[
            #it.supplement #it.counter.display(it.numbering) #it.caption.body
          ]
        ]
      ]
    ]
  }

  // Format Persamaan Matematika (nomor Arab identitas bab di tepi kanan)
  set math.equation(numbering: "(1.1)")

  // ---------------------------------------------------------------------------
  // 3. HALAMAN SAMPUL / COVER RESMI FATISDA UNS
  // ---------------------------------------------------------------------------
  // Catatan Pedoman:
  // (*) Font Times New Roman 12 pt BOLD
  // (**) Font Times New Roman 14 pt BOLD
  // Margin Cover: Atas 4 cm, Kiri 4 cm, Bawah 3 cm, Kanan 3 cm
  page(
    paper: "a4",
    margin: (top: 4cm, left: 4cm, bottom: 3cm, right: 3cm),
    header: none,
    footer: none,
  )[
    #set align(center)
    #set text(font: "Times New Roman")

    #if tipe_dokumen == "proposal" [
      // Format Lampiran 1: Cover Proposal Skripsi
      #text(size: 12pt, weight: "bold")[Proposal Skripsi]
      
      #v(0.8cm)
      #text(size: 14pt, weight: "bold")[#upper(title)]
      
      #v(0.8cm)
      #image("figures/logo_uns.png", width: 4cm)
      
      #v(0.8cm)
      #text(size: 12pt, weight: "bold")[Disusun Oleh :]
      
      #v(0.2cm)
      #text(size: 14pt, weight: "bold")[#upper(author)]
      
      #v(0.1cm)
      #text(size: 12pt, weight: "bold")[#nim]
      
      #v(0.8cm)
      #text(size: 12pt, weight: "bold")[USULAN PENELITIAN]
      
      #v(0.2cm)
      #text(size: 12pt, weight: "bold")[Diajukan untuk menyusun skripsi \ Program Studi #prodi]
      
      #v(0.8cm)
      #text(size: 12pt, weight: "bold")[
        #upper(fakultas) \
        #upper(universitas)
      ]
      
      #v(0.1cm)
      #text(size: 14pt, weight: "bold")[
        #upper(kota) \
        TAHUN #tahun
      ]
    ] else [
      // Format Lampiran 4: Cover Skripsi Final
      #v(0.5cm)
      #text(size: 14pt, weight: "bold")[#upper(title)]
      
      #v(0.5cm)
      #text(size: 12pt, weight: "bold")[SKRIPSI]
      
      #v(0.2cm)
      #text(size: 12pt, weight: "bold")[
        Diajukan untuk memenuhi sebagian persyaratan mendapatkan gelar Strata Satu \
        Program Studi #prodi
      ]
      
      #v(0.8cm)
      #image("figures/logo_uns.png", width: 4cm)
      
      #v(0.8cm)
      #text(size: 12pt, weight: "bold")[Disusun oleh :]
      
      #v(0.2cm)
      #text(size: 14pt, weight: "bold")[#upper(author)]
      
      #v(0.1cm)
      #text(size: 12pt, weight: "bold")[#nim]
      
      #v(0.8cm)
      #text(size: 12pt, weight: "bold")[
        #upper(fakultas) \
        #upper(universitas)
      ]
      
      #v(0.1cm)
      #text(size: 14pt, weight: "bold")[
        #upper(kota) \
        Tahun #tahun
      ]
    ]
  ]

  // ---------------------------------------------------------------------------
  // 4. BAGIAN AWAL (PENOMORAN ROMAWI KECIL DI BAWAH TENGAH)
  // ---------------------------------------------------------------------------
  // Cover dihitung sebagai halaman i, halaman persetujuan mulai dari ii
  set page(
    paper: "a4",
    margin: (top: 4cm, left: 4cm, bottom: 3cm, right: 3cm),
    header: none,
    footer: context {
      align(center)[#text(size: 12pt)[#counter(page).display("i")]]
    }
  )
  counter(page).update(2)

  // --- HALAMAN PERSETUJUAN (Lampiran 2 Hal. 38) ---
  if tipe_dokumen == "proposal" [
    #align(center)[
      #text(size: 14pt, weight: "bold")[PERSETUJUAN]
      
      #v(0.8cm)
      #text(size: 12pt)[
        Proposal Skripsi Mahasiswa : \
        *#upper(author)* \
        *#nim*
      ]
      
      #v(0.5cm)
      #text(size: 12pt)[Dengan judul]
      
      #v(0.3cm)
      #text(size: 12pt, weight: "bold")[#upper(title)]
      
      #v(0.6cm)
      #text(size: 12pt)[Telah disetujui untuk diseminarkan pada tanggal, #tanggal_persetujuan]
      
      #v(0.8cm)
      #text(size: 12pt)[Persetujuan Pembimbing :]
      
      #v(0.6cm)
      #grid(
        columns: (1fr, 1fr),
        align: (center, center),
        [
          Pembimbing I \
          #v(2cm)
          *#pembimbing1.nama* \
          NIP. #pembimbing1.nip
        ],
        [
          #if pembimbing2.nama != "" [
            Pembimbing II \
            #v(2.5cm)
            *#pembimbing2.nama* \
            NIP. #pembimbing2.nip
          ]
        ]
      )
    ]
    #pagebreak()
    
    // --- HALAMAN PENGESAHAN PROPOSAL (Lampiran 3 Hal. 39) ---
    #align(center)[
      #text(size: 14pt, weight: "bold")[PENGESAHAN]
      
      #v(0.5cm)
      #text(size: 12pt)[
        Proposal Skripsi Mahasiswa : \
        *#upper(author)* \
        *#nim*
      ]
      
      #v(0.3cm)
      #text(size: 12pt)[Dengan judul]
      
      #v(0.3cm)
      #text(size: 12pt, weight: "bold")[#upper(title)]
      
      #v(0.4cm)
      #text(size: 12pt)[Telah diseminarkan pada tanggal, #tanggal_seminar]
      
      #v(0.5cm)
      #text(size: 12pt)[Susunan Penguji Seminar Proposal :]
    ]
    
    #v(0.3cm)
    #align(center)[
      #block(width: 85%)[
        #grid(
          columns: (2.5fr, 1.5fr),
          row-gutter: 1.2em,
          [1. #pembimbing1.nama \ #h(0.4cm) NIP. #pembimbing1.nip], [(..........................)],
          [#if pembimbing2.nama != "" [2. #pembimbing2.nama \ #h(0.4cm) NIP. #pembimbing2.nip]], [#if pembimbing2.nama != "" [(..........................)]],
          [#if penguji_utama.nama != "" [3. #penguji_utama.nama \ #h(0.4cm) NIP. #penguji_utama.nip]], [#if penguji_utama.nama != "" [(..........................)]],
        )
      ]
    ]
    
    #v(0.6cm)
    #align(center)[Mengetahui,]
    #v(0.3cm)
    #grid(
      columns: (1fr, 1fr),
      align: (center, center),
      [
        Koordinator Skripsi \
        #v(1.8cm)
        *#koordinator_skripsi.nama* \
        NIP. #koordinator_skripsi.nip
      ],
      [
        Kepala Program Studi \
        #v(1.8cm)
        *#kaprodi.nama* \
        NIP. #kaprodi.nip
      ]
    )
    #pagebreak()
  ]

  // --- HALAMAN MOTTO & PERSEMBAHAN (Opsional) ---
  if motto != none [
    #align(center + horizon)[
      #text(size: 14pt, weight: "bold")[MOTTO]
      #v(1cm)
      #text(style: "italic")[#motto]
    ]
    #pagebreak()
  ]

  if persembahan != none [
    #align(center + horizon)[
      #text(size: 14pt, weight: "bold")[PERSEMBAHAN]
      #v(1cm)
      #text(style: "italic")[#persembahan]
    ]
    #pagebreak()
  ]

  // --- HALAMAN ABSTRAK (INDONESIA) (Lampiran 7a Hal. 46) ---
  if abstrak_id != none [
    #align(center)[
      #text(size: 12pt, weight: "bold")[#upper(title)] \
      #v(0.5cm)
      #text(size: 12pt, weight: "bold")[#upper(author)] \
      #v(0.2cm)
      #text(size: 12pt)[
        Program Studi #prodi #fakultas \
        #universitas
      ]
      #v(0.8cm)
      #text(size: 14pt, weight: "bold")[ABSTRAK]
    ]
    
    #v(0.5cm)
    // Abstrak wajib spasi tunggal (1 spasi) dan alinea menjorok
    #set par(justify: true, leading: 0.65em, first-line-indent: 1.5cm)
    #abstrak_id.body
    
    #v(0.8cm)
    #set par(first-line-indent: 0cm)
    *Kata kunci:* #abstrak_id.keywords
    #pagebreak()
  ]

  // --- HALAMAN ABSTRACT (INGGRIS) (Lampiran 7b Hal. 48) ---
  if abstrak_en != none [
    #align(center)[
      #text(size: 12pt, weight: "bold")[#upper(abstrak_en.title)] \
      #v(0.5cm)
      #text(size: 12pt, weight: "bold")[#upper(author)] \
      #v(0.2cm)
      #text(size: 12pt)[
        Department of Informatics, Faculty of Information Technology and Data Science \
        Universitas Sebelas Maret
      ]
      #v(0.8cm)
      #text(size: 14pt, weight: "bold")[ABSTRACT]
    ]
    
    #v(0.5cm)
    #set par(justify: true, leading: 0.65em, first-line-indent: 1.5cm)
    #abstrak_en.body
    
    #v(0.8cm)
    #set par(first-line-indent: 0cm)
    *Keywords:* #abstrak_en.keywords
    #pagebreak()
  ]

  // --- KATA PENGANTAR (Lampiran 8 Hal. 50) ---
  if kata_pengantar != none [
    #align(center)[
      #text(size: 14pt, weight: "bold")[KATA PENGANTAR]
    ]
    #v(0.8cm)
    #set par(justify: true, leading: 1em, first-line-indent: 1.5cm)
    #kata_pengantar
    
    #v(1.5cm)
    #align(right)[
      #block(width: 40%)[
        #align(center)[
          #kota, #bulan_tahun \
          #v(2cm)
          Penulis
        ]
      ]
    ]
    #pagebreak()
  ]

  // --- DAFTAR ISI, DAFTAR GAMBAR, DAFTAR TABEL (Lampiran 9, 10a, 10b) ---
  if tampilkan_daftar_isi [
    #align(center)[#text(size: 14pt, weight: "bold")[DAFTAR ISI]]
    #v(0.3cm)
    #align(right)[*Halaman*]
    #v(0.2cm)
    #outline(title: none, depth: 3, indent: 1.5em)
    #pagebreak()
  ]

  if tampilkan_daftar_gambar [
    #align(center)[#text(size: 14pt, weight: "bold")[DAFTAR GAMBAR]]
    #v(0.3cm)
    #align(right)[*Halaman*]
    #v(0.2cm)
    #outline(title: none, target: figure.where(kind: image))
    #pagebreak()
  ]

  if tampilkan_daftar_tabel [
    #align(center)[#text(size: 14pt, weight: "bold")[DAFTAR TABEL]]
    #v(0.3cm)
    #align(right)[*Halaman*]
    #v(0.2cm)
    #outline(title: none, target: figure.where(kind: table))
    #pagebreak()
  ]

  // ---------------------------------------------------------------------------
  // 5. BAGIAN UTAMA (PENOMORAN ANGKA ARAB DENGAN ATURAN KHUSUS)
  // ---------------------------------------------------------------------------
  // Aturan Pedoman Hal. 33-34:
  // - Angka Arab dimulai dari 1 di Bab 1.
  // - Posisi penomoran: di KANAN ATAS (3cm dari kanan, 1.5cm dari atas).
  // - KECUALI halaman pertama tiap Judul Bab baru: di BAWAH TENGAH (1.5cm dari bawah).
  set page(
    paper: "a4",
    margin: (top: 4cm, left: 4cm, bottom: 3cm, right: 3cm),
    header: context {
      // Cek apakah halaman saat ini memuat Judul Bab (Heading Level 1)
      let cur_page = here().page()
      let has_chapter = query(heading.where(level: 1)).any(h => h.location().page() == cur_page)
      if not has_chapter {
        align(right)[#text(size: 12pt)[#counter(page).display("1")]]
      }
    },
    footer: context {
      let cur_page = here().page()
      let has_chapter = query(heading.where(level: 1)).any(h => h.location().page() == cur_page)
      if has_chapter {
        align(center)[#text(size: 12pt)[#counter(page).display("1")]]
      }
    }
  )
  counter(page).update(1)

  // Pengaturan Paragraf Naskah Utama (1.5 spasi, indent 1.5cm, justify)
  set par(
    justify: true,
    leading: 1em, // Spasi 1.5 standar Times New Roman
    first-line-indent: 1.5cm,
  )

  // Pengaturan Hirarki Heading (BAB IV Tata Cara Penulisan Hal. 33)
  // Level 1: Judul Bab (Kapital, 14pt Bold, Center, Halaman Baru)
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(0.5cm) // Jarak 4 cm dari tepi atas naskah
    align(center)[
      #text(size: 14pt, weight: "bold")[#upper(it.body)]
    ]
    v(1.5em)
  }

  // Level 2: Sub-bab (Title Case, 12pt Bold, Tanpa Titik di Akhir)
  show heading.where(level: 2): it => {
    v(1.2em)
    text(size: 12pt, weight: "bold")[
      #counter(heading).display() #it.body
    ]
    v(0.8em)
  }

  // Level 3: Anak Sub-bab (Title Case, 12pt Italic, Tanpa Titik di Akhir)
  show heading.where(level: 3): it => {
    v(1em)
    text(size: 12pt, weight: "regular", style: "italic")[
      #counter(heading).display() #it.body
    ]
    v(0.6em)
  }

  // Rincian (List): Hanya Angka dan Huruf (Tanpa bullet)
  set enum(indent: 1.5cm, body-indent: 0.5cm)

  // ---------------------------------------------------------------------------
  // 6. RENDER KONTEN ISI DOKUMEN
  // ---------------------------------------------------------------------------
  body
}
