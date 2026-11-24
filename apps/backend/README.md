# Box Archive — Backend API Service

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Gemini--Flash--2.5-orange?style=for-the-badge)](https://python.langchain.com/)
[![Supabase](https://img.shields.io/badge/Supabase-pgvector-green?style=for-the-badge&logo=supabase)](https://supabase.com/)
[![MQTT](https://img.shields.io/badge/MQTT-Paho-purple?style=for-the-badge&logo=eclipse-mosquitto)](https://mqtt.org/)

Backend API untuk **Box Archive**, sistem cerdas Smartclass dan pengarsipan kegiatan belajar mengajar berbasis AI dan IoT. Sistem ini mengotomatisasi manajemen daya kelas, mentranskripsikan dan meringkas rekaman perkuliahan, menyediakan *RAG Chatbot*, serta mendistribusikan materi akademik adaptif.

---

## 📌 Daftar Isi

- [Panduan Instalasi & Setup Lokal](#panduan-instalasi--setup-lokal)
- [Variabel Lingkungan (.env)](#variabel-lingkungan-env)
- [Spesifikasi Endpoint API](#spesifikasi-endpoint-api)
- [Alur Kerja (Workflow Integrasi)](#alur-kerja-workflow-integrasi)
- [Anggota Tim Capstone](#anggota-tim-capstone)

---

## Panduan Instalasi & Setup Lokal

### Prerequisites
- Python >= 3.10
- PostgreSQL / Supabase account
- MQTT Broker (Mosquitto / Cloud MQTT Broker)

### Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/CAZP01/box-archive/
   cd apps/backend
   ```

2. **Buat & Aktifkan Virtual Environment**
   ```bash
   python -m venv venv
   # Linux / macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install Dependensi**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Konfigurasi Environment Variable**
   Salin berkas `.env.example` ke `.env` dan isi sesuai kredensial Anda:
   ```bash
   cp .env.example .env
   ```

5. **Jalankan Server Development**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. **Akses Dokumentasi API Interactive**
   Buka di browser:
   - **Swagger UI**: `http://localhost:8000/docs`
   - **ReDoc**: `http://localhost:8000/redoc`

---

## Variabel Lingkungan (`.env`)

```env
PROJECT_NAME="Box Archive Backend API"
VERSION="1.0.0"

# Database & Vector Store
DATABASE_URL=""
SUPABASE_URL=""
SUPABASE_KEY=""

# AI / Gemini API
GEMINI_API_KEY=""

# MQTT Broker Configuration
MQTT_BROKER_HOST=""
MQTT_BROKER_PORT=1883

# Security JWT
JWT_SECRET_KEY=""
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

---

## Spesifikasi Endpoint API

| Method | Endpoint | Deskripsi | Proteksi |
|---|---|---|---|
| `GET` | `/` | Health check service status | Public |
| `POST` | `/api/v1/auth/login` | Login user & dapatkan token JWT | Public |
| `POST` | `/api/v1/iot/room-status` | Melacak kehadiran dosen & kontrol otomatis alat IoT | Service/API |
| `POST` | `/api/v1/iot/control-device` | Manual control perangkat kelas (AC, Proyektor, Lampu) | Admin |
| `GET` | `/api/v1/archives/adaptive-content` | Mengambil materi adaptif berdasarkan status presensi siswa | JWT User |
| `POST` | `/api/v1/chatbot/query` | Tanya jawab RAG Chatbot seputar materi transkrip kelas | JWT User |

---

## Alur Kerja (Workflow Integrasi)

1. **Awal Kelas**: Face Recognition mendeteksi dosen di kelas + sistem memvalidasi jadwal.
2. **Pengaktifan Fasilitas**: Sinyal dikirim ke `/api/v1/iot/room-status`, FastAPI mempublikasikan perintah `ON` via MQTT ke mikrokontroler (ESP32) untuk menyalakan AC, Lampu, dan Proyektor, serta memulai perekaman audio.
3. **Selesai Kelas**: Sinyal `OFF` dikirim. Perekaman dihentikan, Faster-Whisper membuat transkrip audio, dan LangChain merangkumnya.
4. **Vector Store Ingestion**: Transkrip disimpan ke `pgvector` Supabase.
5. **Post-Class Interaction**:
   - Siswa mengakses PWA untuk membaca ringkasan.
   - Chatbot RAG melayani pertanyaan siswa berdasar transkrip.
   - Siswa Hadir dapat rekomendasi pengayaan; Siswa Absen mendapat materi pendukung dasar.