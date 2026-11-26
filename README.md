# Box Archive: Sistem Smartclass dan Arsip Pencatat KBM di Kelas

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: React](https://img.shields.io/badge/Frontend-React.js-61DAFB?logo=react)](https://reactjs.org/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![AI Integration: LangChain](https://img.shields.io/badge/AI-LangChain%20%2B%20Gemini%202.5-121011?logo=google)](https://python.langchain.com/)
[![IoT Protocol: MQTT](https://img.shields.io/badge/IoT-MQTT%20%2B%20ESP32-660099?logo=eclipse-mosquitto)](https://mqtt.org/)

**Box Archive** adalah ekosistem smartclass berbasis IoT dan Artificial Intelligence (AI) yang dirancang untuk mengotomatisasi efisiensi operasional ruang kelas serta mencatat, meringkas, dan mendistribusikan materi pembelajaran secara adaptif.

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
1. **Pemborosan Energi Fasilitas Kelas:** Perangkat elektronik (AC, lampu, proyektor) sering tetap menyala meskipun ruangan dalam kondisi kosong atau perkuliahan telah selesai.
2. **Hilangnya Pembahasan Verbal:** Penjelasan verbal pengajar, tanya jawab, dan diskusi mendalam saat jam perkuliahan menguap begitu saja tanpa terdokumentasi dengan baik.
3. **Kesenjangan Pemahaman Siswa:** Distribusi materi pasca-kelas umumnya disaratkan sama untuk seluruh siswa tanpa memperhitungkan status kehadiran.

### Solusi **Box Archive**:
Sistem terintegrasi yang memantau keberadaan penanggung jawab kelas/jadwal secara real-time via *Face Recognition* & IoT, mengaktifkan otomatisasi sakelar perangkat keras, merekam interaksi suara kelas, mengonversinya menjadi transkrip terstruktur via AI (Speech-to-Text & LLM), serta menyajikan fitur RAG Chatbot dan rekomendasi materi yang dipersonalisasi berdasarkan variabel presensi.

---

## Fitur Utama

- **Automated Smart Class Trigger (IoT & Vision):** Pengendalian perangkat fisik kelas (lampu, AC, proyektor) via mikrokontroler (ESP32) dan deteksi presensi penanggung jawab kelas via *DeepFace*.
- **Speech-to-Text & Automatic Summarization:** Transkripsi otomatis perkuliahan Bahasa Indonesia menggunakan *Faster-Whisper* dan ekstraksi poin penting/ringkasan terstruktur.
- **Contextual RAG Chatbot:** Fitur tanya-jawab materi perkuliahan cerdas berbasis *LangChain* dan *Gemini 2.5 Flash* yang dibatasi (*guardrailed*) hanya menjawab sesuai konteks materi hari itu.
- **Adaptive Content Distribution:**
  - **Siswa Hadir:** Mengakses transkrip penuh, RAG Chatbot, serta modul *Rekomendasi Pembelajaran Lanjutan*.
  - **Siswa Absen:** Mengakses transkrip/ringkasan, RAG Chatbot, serta penyesuaian *Materi Pendukung Dasar* untuk mengejar ketertinggalan.
- **Data Privacy & Security:** Penghapusan otomatis file rekaman suara mentah (*raw audio*) setelah proses transkripsi selesai untuk menjaga privasi.

---

## Arsitektur Sistem

 visualisasi alur komunikasi antar modul dalam sistem Box Archive:

```
[ Sensor / Mikrokontroler (ESP32) ]
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
| **Frontend** | React.js (PWA) | Progressive Web Application responsif untuk Desktop & Mobile |
| **Backend API** | Python (FastAPI) | Framework REST API tinggi performa & orkestrator layanan |
| **IoT Hardware & Protocol**| ESP32, INMP441, MQTT | Sensor fisik kelas, mikrofon I2S, dan protokol pesan Publish/Subscribe |
| **Computer Vision** | DeepFace + OpenCV | Pipeline pengenalan wajah dosen real-time |
| **Speech Processing** | OpenAI Whisper / Faster-Whisper | Pemrosesan audio perkuliahan Bahasa Indonesia ke Teks |
| **LLM & Agent Framework** | LangChain + Gemini 2.5 Flash | Pembuat ringkasan terstruktur dan mesin AI Chatbot |
| **Database & Vector Store**| Supabase + pgvector / ChromaDB | Penyimpanan relational database & vector embedding data teks |

---

## Persyaratan Sistem & Fitur Non-Fungsional

- **Performa Antarmuka:** Penguatan respons Web PWA $< 3-5$ detik pada koneksi internet standar.
- **Batas Waktu Pemrosesan AI:** Konversi audio ke transkrip dan ringkasan diselesaikan maksimal $30$ menit setelah sesi kelas ditutup.
- **AI Safety & Hallucination Prevention:** Chatbot secara ketat mengembalikan pesan *fallback*: `"Maaf, informasi ini tidak terdapat dalam rekaman kelas hari ini"` jika query berada di luar konteks transkrip.
- **Privasi File Audio:** File *raw audio* dihapus otomatis dari server lokal/cloud setelah transkripsi berhasil disimpan ke basis data.

---

## Tim & Pembagian Peran

| NIM | Nama | Peran & Tanggung Jawab Utama |
|---|---|---|
| **23106050024** | Chairul ‘Azmi Zuhdi Pramono | Lead Backend Engineer API |
| **23106050016** | M. Alan Daulay | Speech-to-Text & Summarization (Audio & NLP Pipeline) & LangChain Agent Architect |
| **23106050065** | Shofy Naila Az-Zahra | Frontend Engineer (Progressive Web App - React) |
| **23106050020** | Aisyah Ayudia Inara | Database Administrator & Vector DB Setup (Supabase/pgvector) |
| **23106050046** | Dwi Estriana | Implementasi Komunikasi MQTT Broker, Publisher & Subscriber |
| **23106050057** | Galih Fathurahman Ardiansyah | Face Recognition Pipeline (DeepFace & OpenCV) & LangChain Agent Architect |
| **23106050087** | Sidhi Akmal Mawla | IoT Hardware Engineer & Sensor Data Publisher |
| - | Anggraini Wijayanti | IoT Service Integration |

---

## Struktur Proyek

```bash
box-archive/
├── apps/
│   ├── web/                   # Frontend React.js PWA
│   └── backend/               # FastAPI Server & REST API Endpoints
├── services/
│   ├── ai_pipeline/           # Faster-Whisper STT & DeepFace Vision Service
│   ├── langchain_agent/       # LangChain Agent, Prompts, & RAG Chains
│   └── iot_service/           # MQTT Publisher, Subscriber, & Device Handler
├── hardware/
│   └── esp32_firmware/        # Firmware C++/Arduino untuk Mikrokontroler ESP32
├── docs/                      # Dokumentasi API, Use Case, & Laporan Capstone
├── docker-compose.yml         # Konfigurasi container lokal
└── README.md                  # Dokumentasi Utama
```

---

## Panduan Instalasi & Pengoperasian

### 1. Prasyarat Environment
- Python `3.10+`
- Node.js `v18+` & npm/yarn
- MQTT Broker (e.g., Mosquitto)
- Supabase Account / PostgreSQL dengan ekstensi `pgvector`

### 2. Setup Backend API
```bash
# Clone Repository
git clone https://github.com/organization/box-archive.git
cd box-archive/apps/backend

# Buat dan aktifkan Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/macOS atau `venv\Scripts\activate` di Windows

# Install Dependensi
pip install -r requirements.txt

# Jalankan FastAPI Server
uvicorn main:app --reload --port 8000
```

### 3. Setup Frontend (PWA)
```bash
cd ../../apps/web

# Install dependensi frontend
npm install

# Jalankan server pengembangan lokal
npm run dev
```

### 4. Setup Service AI & Agent
```bash
cd ../../services/langchain_agent
# Salin konfigurasi environment
cp .env.example .env
# Isikan API Key Gemini dan credentials database di file .env
```

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan Capstone Project Software Engineering. Hak Cipta (c) 2026 Tim Development Box Archive.