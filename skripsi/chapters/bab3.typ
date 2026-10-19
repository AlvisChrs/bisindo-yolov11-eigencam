= BAB 3 METODOLOGI PENELITIAN

== Pengumpulan dan Persiapan Dataset
Dataset dikumpulkan melalui rekaman gestur alfabet BISINDO dari berbagai sudut pandang dan kondisi pencahayaan. Total dataset mencakup 8.743 gambar dengan 9.226 anotasi objek yang terbagi atas split data latih (train), validasi (val), dan uji (test).

== Evaluasi Kinerja Model
Metrik evaluasi yang digunakan meliputi *Mean Average Precision* (mAP):

$ "mAP" = 1/n sum_(k=1)^n "AP"_k $
