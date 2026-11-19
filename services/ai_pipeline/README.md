# AI Pipeline

Modul ini memproses audio manual atau rekaman kelas menjadi arsip teks:

1. Audio ditranskripsikan ke Bahasa Indonesia menggunakan Speech-to-Text.
2. Transkrip Bahasa Indonesia diringkas menjadi tiga segmen: poin utama materi, sesi tanya jawab, dan tugas. Modul memakai Gemini 2.5 Flash jika `GEMINI_API_KEY` tersedia; tanpa API key, modul menggunakan ekstraksi kata kunci lokal.
3. Pemrosesan diselesaikan maksimal 30 menit setelah kelas ditutup.
4. Audio mentah dihapus otomatis setelah transkripsi berhasil dan teks utuh tersedia; jika transkripsi gagal, audio dipertahankan agar dapat dicoba ulang.

Konfigurasi lokal dapat disiapkan dengan menyalin `.env.example` ke `.env`.

Jalankan pipeline dari folder ini dengan `python main_pipeline.py path/ke/audio.mp3`.
Hasil JSON disimpan sebagai `output_hasil_kelas.json` di folder ini secara default.
Gunakan opsi `--output path/hasil.json` untuk memilih lokasi lain.
