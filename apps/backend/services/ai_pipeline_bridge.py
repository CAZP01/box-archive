"""
Jembatan integrasi antara FastAPI Backend (`apps/backend`) dan modul
Speech-to-Text & Summarization (`services/ai_pipeline`).

Menjalankan Proses 3.0 (FR-04, FR-05, NFR-05, NFR-07):
1. Menerima file rekaman audio kelas sementara dari endpoint FastAPI.
2. Memanggil `services.ai_pipeline.run_pipeline(...)`.
3. Menyimpan hasil arsip (transkrip utuh, segmen ber-timestamp, dan ringkasan 3 segmen)
   ke penyimpanan arsip (D3 Arsip) yang siap dibaca oleh endpoint Archives dan RAG Chatbot.
4. Memastikan file audio mentah otomatis terhapus hanya setelah arsip tersimpan.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Tambahkan root repositori `box-archive/` ke sys.path agar `services.ai_pipeline` dapat diimpor
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.ai_pipeline.main_pipeline import run_pipeline

ARCHIVE_STORAGE_DIR = REPO_ROOT / "apps" / "backend" / "storage" / "archives"
TEMP_AUDIO_DIR = REPO_ROOT / "apps" / "backend" / "storage" / "temp_audio"

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_lecture_id(lecture_id: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", lecture_id.strip()).strip("_")
    if not cleaned:
        raise ValueError("lecture_id tidak boleh kosong.")
    return cleaned


def get_archive_path(lecture_id: str) -> Path:
    """Kembalikan lokasi file JSON arsip untuk satu sesi perkuliahan."""
    safe_id = _sanitize_lecture_id(lecture_id)
    return ARCHIVE_STORAGE_DIR / f"{safe_id}.json"


def process_uploaded_audio(
    lecture_id: str,
    temp_audio_path: Path,
    *,
    course_name: str | None = None,
    delete_audio: bool = True,
    allow_fallback_on_error: bool = False,
) -> dict[str, Any]:
    """
    Proses file audio kelas melalui `services/ai_pipeline` dan kembalikan payload arsip.

    Sesuai FR-05 & NFR-05:
    - File `temp_audio_path` otomatis dihapus oleh `run_pipeline` HANYA SETELAH
      file arsip JSON berhasil disimpan ke `ARCHIVE_STORAGE_DIR`.
    - Jika proses gagal, `temp_audio_path` tetap dipertahankan untuk retry.
    """
    safe_id = _sanitize_lecture_id(lecture_id)
    ARCHIVE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = get_archive_path(safe_id)

    saved_path = run_pipeline(
        audio_path=temp_audio_path,
        output_path=archive_path,
        session_id=safe_id,
        delete_audio=delete_audio,
        allow_fallback_on_error=allow_fallback_on_error,
        show_progress=False,
    )

    payload: dict[str, Any] = json.loads(saved_path.read_text(encoding="utf-8"))
    if course_name:
        payload["course_name"] = course_name
        saved_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def load_archive_by_lecture(lecture_id: str) -> dict[str, Any] | None:
    """Muat data arsip (transkrip + ringkasan 3 segmen) berdasarkan lecture_id."""
    archive_path = get_archive_path(lecture_id)
    if not archive_path.is_file():
        return None
    return json.loads(archive_path.read_text(encoding="utf-8"))


def load_transcript_text(lecture_id: str) -> str | None:
    """Ambil teks transkrip penuh untuk kebutuhan RAG Chatbot (Proses 4.0)."""
    archive = load_archive_by_lecture(lecture_id)
    if not archive:
        return None
    transcript = str(archive.get("transkrip", "")).strip()
    return transcript or None
