# Box Archive: Sistem Smartclass dan Arsip Pencatat Kegiatan Belajar Mengajar di Kelas

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: React](https://img.shields.io/badge/Frontend-React.js-61DAFB?logo=react)](https://reactjs.org/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![AI Integration: LangChain](https://img.shields.io/badge/AI-LangChain%20%2B%20Gemini%202.5-121011?logo=google)](https://python.langchain.com/)
[![IoT Protocol: MQTT](https://img.shields.io/badge/IoT-MQTT%20%2B%20ESP32-660099?logo=eclipse-mosquitto)](https://mqtt.org/)

**Box Archive** adalah ekosistem *smartclass* berbasis Internet of Things (IoT) dan Artificial Intelligence (AI) yang dirancang untuk mengotomatisasi efisiensi perangkat ruang kelas serta merekam, mentranskripsikan, meringkas, dan mendistribusikan arsip materi pembelajaran secara adaptif berdasarkan kehadiran mahasiswa.

---

## Daftar Isi
- [Latar Belakang & Masalah](#latar-belakang--masalah)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Teknologi yang Digunakan (Tech Stack)](#teknologi-yang-digunakan-tech-stack)
- [Persyaratan Sistem & Fitur Non-Fungsional](#persyaratan-sistem--fitur-non-fungsional)
- [Tim & Pembagian Peran](#tim--pembagian-peran)
- [Struktur Proyek](#struktur-proyek)
- [Panduan Instalasi & Pengoperasian](#panduan-instalasi--pengoperasian)

---

## Latar Belakang & Masalah

### Masalah yang Dihadapi:
1. **Pemborosan Energi Fasilitas Kelas:** Perangkat elektronik (AC, lampu, proyektor) sering tetap menyala meskipun ruangan kosong atau tidak ada mata kuliah yang berlangsung.
2. **Hilangnya Pembahasan Verbal:** Penjelasan lisan pengajar, sesi tanya jawab, dan diskusi mendalam saat perkuliahan hilang setelah kelas selesai karena hanya mengandalkan salinan presentasi atau catatan manual.
3. **Kesenjangan Pemahaman Mahasiswa:** Sistem akademik umumnya memberikan akses rekapitulasi materi yang sama kepada seluruh mahasiswa tanpa memandang status kehadiran mereka di kelas.

### Solusi **Box Archive**:
Sistem terintegrasi yang memvalidasi jadwal dan mendeteksi keberadaan dosen secara *real-time* via *Face Recognition* & IoT untuk menyalakan/mematikan perangkat elektronik kelas secara otomatis, merekam interaksi suara kelas, mengonversinya menjadi transkrip dan rangkuman 3 segmen (*poin utama materi*, *sesi tanya jawab*, *tugas*) via *Speech-to-Text* & LLM, serta mendistribusikan materi tindak lanjut secara adaptif beserta RAG Chatbot sesuai status kehadiran mahasiswa.

---

## Fitur Utama

- **Otomasi Kelas & Kontrol Perangkat (FR-01 – FR-03):** Validasi jadwal *real-time* dan deteksi kehadiran dosen via *DeepFace* untuk menyalakan perangkat elektronik kelas (AC, lampu, proyektor), memulai perekaman audio saat status kelas `Active`, serta mematikan perangkat saat kelas ditutup.
- **Speech-to-Text & Peringkasan Terstruktur 3 Segmen (FR-04):** Transkripsi otomatis rekaman suara kelas Bahasa Indonesia menggunakan *Faster-Whisper* dan peringkasan menggunakan *LangChain + Gemini 2.5 Flash* ke dalam tiga segmen: **Poin Utama Materi**, **Sesi Tanya Jawab**, dan **Tugas**.
- **Privasi & Integritas Penghapusan Audio (FR-05 & NFR-05):** Penghapusan otomatis rekaman suara mentah (*raw audio*) dari server segera setelah transkrip dan rangkuman berhasil tersimpan, dengan mekanisme retensi sementara untuk pemrosesan ulang (*retry*) apabila terjadi kegagalan proses.
- **Distribusi Materi Adaptif Berbasis Kehadiran (FR-06 – FR-08):**
  - **Mahasiswa Hadir:** Mengakses transkrip penuh, rangkuman terstruktur, RAG Chatbot, serta *Rekomendasi Bacaan Lanjutan*.
  - **Mahasiswa Absen:** Mengakses rangkuman terstruktur, RAG Chatbot, serta distribusi *Materi Pendukung Dasar* untuk mengejar ketertinggalan konteks.
- **Contextual RAG Chatbot (NFR Reliability):** Chatbot interaktif berbasis *LangChain* dan *Gemini 2.5 Flash* yang dibatasi (*guardrailed*) hanya menjawab berdasarkan konteks transkrip perkuliahan hari itu.

---

## Arsitektur Sistem

```text
[ Sensor / Mikrokontroler (ESP32 + INMP441 I2S) ]
          | (MQTT Protocol)
          v
[ Broker MQTT / Cloud ] ---> [ FastAPI Backend ] ---> [ Database Utama (Supabase) ]
                                   |
    +------------------------------+------------------------------+
    |                              |                              |
    v                              v                              v
[ Face Recognition ]      [ Faster-Whisper ]           [ LangChain + Gemini ]
(Melacak Ruangan)          (Audio -> Teks)             (Summarization & RAG)
                                                                  |
                                                                  v
                                                        [ pgvector / ChromaDB ]
                                                                  |
                                                                  v
                                                             [ PWA App ]
```

---

## Teknologi yang Digunakan (Tech Stack)

| Kategori | Teknologi | Deskripsi |
|---|---|---|
| **Frontend** | React.js (PWA) | Progressive Web Application responsif untuk PC/Laptop & Mobile |
| **Backend API** | Python (FastAPI) | REST API server, orkestrasi bisnis, dan integrasi layanan |
| **IoT Hardware & Protocol** | ESP32, INMP441 I2S, MQTT | Mikrokontroler kelas, mikrofon digital I2S, dan komunikasi Publish/Subscribe |
| **Computer Vision** | DeepFace + OpenCV | Pipeline pengenalan wajah dosen dan deteksi kehadiran *real-time* |
| **Speech-to-Text & NLP** | Faster-Whisper + LangChain + Gemini 2.5 Flash | Transkripsi audio Bahasa Indonesia dan ekstraksi rangkuman 3 segmen |
| **RAG Agent & Chatbot** | LangChain + Gemini 2.5 Flash | Mesin tanya-jawab kontekstual dan distribusi rekomendasi materi |
| **Database & Vector Store** | Supabase + pgvector / ChromaDB | Basis data relasional utama dan penyimpanan *vector embedding* arsip |

---

## Persyaratan Sistem & Fitur Non-Fungsional

- **Kinerja Antarmuka:** Aplikasi web memuat data dalam waktu kurang dari 3–5 detik pada koneksi internet standar.
- **Batas Waktu Pemrosesan AI:** Konversi suara ke teks (STT) dan peringkasan 3 segmen diselesaikan maksimal **30 menit** setelah kelas resmi ditutup oleh dosen di web app.
- **Keandalan Chatbot (Anti-Halusinasi):** Chatbot wajib memberikan respons baku `"Maaf, informasi ini tidak terdapat dalam rekaman kelas hari ini"` jika mahasiswa bertanya di luar konteks transkrip kelas.
- **Privasi Audio (FR-05 & NFR-05):** Data rekaman suara mentah (*raw audio*) dihapus otomatis dari server setelah transkrip dan rangkuman berhasil tersimpan.

---

## Tim & Pembagian Peran

Sesuai dengan pembagian tugas pada Proposal Capstone Box Archive:

| NIM | Nama | Tugas dalam Proyek |
|---|---|---|
| **23106050024** | Chairul ‘Azmi Zuhdi Pramono | Menyusun backend dan LangChain agent |
| **23106050016** | M. Alan Daulay | *Speech-to-Text & Summarization* (Audio & NLP Pipeline) |
| **23106050065** | Shofy Naila Az-Zahra | Membangun *Progressive Web App* (PWA) |
| **23106050020** | Aisyah Ayudia Inara | Menyusun Database |
| **23106050046** | Dwi Estriana | Implementasi komunikasi MQTT antara perangkat IoT dan server menggunakan MQTT Broker, Publisher, dan Subscriber |
| **23106050057** | Galih Fathurahman Ardiansyah | Membangun pipeline *Face Recognition* (DeepFace & OpenCV) untuk deteksi wajah dosen secara *real-time* |
| **23106050087** | Sidhi Akmal Mawla | Membangun hardware IoT dan Publisher data sensor MQTT |
| - | Anggraini Wijayanti | IoT Service |

---

## Struktur Proyek

```text
box-archive/
├── apps/                      # Aplikasi Frontend (PWA) & Backend API (FastAPI)
├── services/
│   └── ai_pipeline/           # Modul Speech-to-Text (Faster-Whisper) & Summarization 3 Segmen
├── hardware/                  # Firmware ESP32, sensor ruangan, & konfigurasi mikrofon INMP441 I2S
└── README.md                  # Dokumentasi utama proyek
```

---

## Panduan Instalasi & Pengoperasian

### 1. Prasyarat Environment
- Python `3.10+`
- Node.js `v18+` & npm
- MQTT Broker (Mosquitto / Cloud Broker)
- Supabase / PostgreSQL dengan ekstensi `pgvector`

### 2. Menjalankan Modul Speech-to-Text & Summarization (`services/ai_pipeline`)
```powershell
cd services/ai_pipeline
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Salin konfigurasi environment dan isi GEMINI_API_KEY
Copy-Item .env.example .env

# Jalankan pipeline pada file rekaman audio kelas
python main_pipeline.py path/ke/rekaman_kelas.mp3
```

---

## Lisensi

Proyek ini dikembangkan untuk keperluan Capstone Project. Hak Cipta © 2026 Tim Pengembang Box Archive.
