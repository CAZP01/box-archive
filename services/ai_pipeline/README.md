# AI Pipeline — Speech-to-Text & Summarization (Proses 3.0)

**Penanggung Jawab:** M. Alan Daulay (23106050016)  
**Peran Proposal:** *Speech-to-Text & Summarization (Audio & NLP Pipeline)*

Modul ini mengimplementasikan **Proses 3.0 (Rekam dan Proses Arsip)** pada sistem **Box Archive** sesuai spesifikasi Proposal Capstone:

1. **Speech-to-Text Bahasa Indonesia (`stt_processor.py` — FR-04 & NFR-07):**
   Mengonversi rekaman audio kelas (`.mp3`, `.wav`, `.m4a`, `.webm`, `.ogg`, `.flac`) menjadi transkrip teks Bahasa Indonesia menggunakan **Faster-Whisper** dengan *Voice Activity Detection* (VAD), *initial prompt* akademik Bahasa Indonesia, dan pembersihan karakter anomali untuk meminimalkan *Word Error Rate* (WER).
2. **Summarization 3 Segmen Terstruktur (`summarizer.py` — FR-04 & NFR-07):**
   Meringkas transkrip perkuliahan menggunakan **LangChain + Gemini 2.5 Flash** (`gemini-2.5-flash`) dengan validasi skema **Pydantic** ke dalam tepat tiga segmen:
   - `poin_utama_materi`: Poin-poin inti materi yang dibahas dosen.
   - `sesi_tanya_jawab`: Pertanyaan dan jawaban/diskusi materi selama kelas.
   - `tugas`: Instruksi tugas, latihan, proyek, atau tenggat pengumpulan yang diberikan dosen.
   *(Apabila `GEMINI_API_KEY` belum diisi, modul menyediakan ekstraktor NLP lokal berbasis pembobotan kata kunci akademik sebagai fallback).*
3. **Batas Waktu Kinerja & Retry Otomatis (`main_pipeline.py` — NFR Performance & NFR-08):**
   Memantau total durasi pemrosesan agar selesai dalam batas maksimal **30 menit (1800 detik)** setelah kelas ditutup, dilengkapi mekanisme *retry* otomatis saat memanggil layanan LLM.
4. **Integritas Penghapusan Audio Mentah (FR-05 & NFR-05):**
   File rekaman suara mentah (*raw audio*) **hanya dihapus setelah** transkrip dan rangkuman 3 segmen berhasil disimpan secara utuh ke penyimpanan arsip. Apabila proses transkripsi, peringkasan, atau penyimpanan gagal, file audio mentah **tetap dipertahankan** agar dapat diproses ulang (*retry*).

---

## Struktur File Modul

```text
services/ai_pipeline/
├── .env.example          # Template konfigurasi environment (API Key, ukuran model, batas waktu)
├── .gitignore            # Daftar file lokal/sementara yang diabaikan Git
├── README.md             # Dokumentasi modul AI Pipeline
├── requirements.txt      # Dependensi Python (faster-whisper, langchain, pydantic, dll.)
├── stt_processor.py      # Modul Speech-to-Text (Faster-Whisper Bahasa Indonesia)
├── summarizer.py         # Modul Summarization 3 Segmen (LangChain + Gemini 2.5 Flash)
├── main_pipeline.py      # Orkestrator pipeline, penyimpanan arsip, & penghapusan audio
└── evaluate_wer.py       # Skrip evaluasi akurasi WER, CER, & RTF (NFR-07)
```

---

## Instalasi & Konfigurasi

1. Buat dan aktifkan *virtual environment* Python (`3.10+`):
   ```powershell
   cd services/ai_pipeline
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Salin `.env.example` menjadi `.env` dan isi `GEMINI_API_KEY`:
   ```powershell
   Copy-Item .env.example .env
   ```

---

## Cara Menjalankan

### 1. Melalui Command Line (CLI)

- **Mode Produksi (Default — Menghapus file audio mentah setelah arsip tersimpan sesuai FR-05/NFR-05):**
  ```powershell
  python main_pipeline.py path/ke/rekaman_kelas.mp3 --session-id sesi-01
  ```
- **Mode Uji Coba / Development (Mempertahankan file audio sampel dengan `--keep-audio`):**
  ```powershell
  python main_pipeline.py path/ke/sample_audio.mp3 --output hasil_uji.json --keep-audio
  ```

### 2. Dipanggil dari Backend FastAPI

```python
from pathlib import Path
from services.ai_pipeline.main_pipeline import run_pipeline

output_json_path = run_pipeline(
    audio_path=Path("storage/temp_audio/sesi_101.wav"),
    output_path=Path("storage/archives/sesi_101.json"),
    session_id="sesi_101",
    delete_audio=True,  # True = hapus raw audio setelah sukses disimpan (FR-05 & NFR-05)
)
```

### 3. Evaluasi Akurasi STT (WER, CER, & RTF — NFR-07)

Gunakan `evaluate_wer.py` untuk menguji tingkat kesalahan kata (*Word Error Rate*), tingkat kesalahan karakter (*Character Error Rate*), serta *Real-Time Factor* (RTF) guna melengkapi pengujian **NFR-07**:

- **Evaluasi dari file JSON hasil `main_pipeline.py` (atau file `.txt`):**
  ```powershell
  python evaluate_wer.py --reference ground_truth.txt --hypothesis output_hasil_kelas.json --target-wer 20
  ```
- **Benchmark perbandingan beberapa ukuran model Whisper pada satu audio uji (tanpa menghapus audio):**
  ```powershell
  python evaluate_wer.py --audio sample_uji.wav --reference ground_truth.txt --compare-models tiny base small --target-wer 20 --output-json eval_report.json
  ```

---

## Format Keluaran JSON (Input untuk D3 Arsip & Proses 4.0)

```json
{
  "session_id": "sesi-01",
  "metadata": {
    "timestamp_selesai_utc": "2026-11-26T16:40:00+00:00",
    "durasi_audio_detik": 5899.4,
    "waktu_stt_detik": 540.12,
    "waktu_summarization_detik": 4.85,
    "waktu_pemrosesan_detik": 545.0,
    "memenuhi_target_30_menit": true,
    "target_detik": 1800,
    "bahasa": "id",
    "probabilitas_bahasa": 0.9985,
    "jumlah_segmen_transkrip": 1240,
    "status_audio_mentah": "dihapus_setelah_tersimpan",
    "model": {
      "speech_to_text": "faster-whisper/base",
      "summarization": "gemini-2.5-flash"
    }
  },
  "transkrip": "Teks transkrip utuh seluruh perkuliahan...",
  "segmen_transkrip": [
    {
      "start": 0.0,
      "end": 5.4,
      "text": "Kalimat segmen pertama..."
    }
  ],
  "ringkasan": {
    "poin_utama_materi": [
      "Poin utama materi 1...",
      "Poin utama materi 2..."
    ],
    "sesi_tanya_jawab": [
      "Pertanyaan: ... | Jawaban: ..."
    ],
    "tugas": [
      "Instruksi tugas mahasiswa..."
    ]
  }
}
```
