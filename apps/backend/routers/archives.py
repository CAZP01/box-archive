from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from schema.archive import (
    AdaptiveMaterialOut,
    ArchiveDetailOut,
    ArchiveProcessResponse,
    StructuredSummary,
)
from services.ai_pipeline_bridge import (
    TEMP_AUDIO_DIR,
    load_archive_by_lecture,
    process_uploaded_audio,
)

router = APIRouter(prefix="/api/v1/archives", tags=["Archives & Adaptive Distribution"])


@router.post("/process-audio", response_model=ArchiveProcessResponse)
async def process_class_audio_endpoint(
    lecture_id: str = Form(..., description="ID sesi perkuliahan (D2 Sesi kelas)"),
    course_name: str | None = Form(None, description="Nama mata kuliah (opsional)"),
    audio_file: UploadFile = File(
        ..., description="File rekaman audio kelas (.wav, .mp3, .m4a, .webm)"
    ),
):
    """
    Endpoint Proses 3.0 (FR-04, FR-05, NFR-05):
    - Menerima file rekaman audio kelas via HTTP multipart saat kelas ditutup (FR-11).
    - Menjalankan Speech-to-Text (Faster-Whisper) dan Summarization 3 Segmen (Gemini 2.5 Flash).
    - Menyimpan hasil ke D3 Arsip dan menghapus file audio mentah secara otomatis setelah tersimpan.
    """
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"
    temp_audio_path = TEMP_AUDIO_DIR / f"{lecture_id}_{uuid.uuid4().hex}{suffix}"

    try:
        with temp_audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    except Exception as exc:
        temp_audio_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Gagal menyimpan file audio sementara: {exc}"
        ) from exc
    finally:
        await audio_file.close()

    try:
        payload = process_uploaded_audio(
            lecture_id=lecture_id,
            temp_audio_path=temp_audio_path,
            course_name=course_name,
            delete_audio=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Sesuai NFR-05 & NFR-08: audio mentah dipertahankan sementara untuk retry jika gagal
        raise HTTPException(
            status_code=500,
            detail=(
                f"Pemrosesan AI Pipeline gagal. Audio sementara dipertahankan di server "
                f"untuk diproses ulang (NFR-05): {exc}"
            ),
        ) from exc

    return ArchiveProcessResponse(
        status="success",
        lecture_id=payload["session_id"],
        course_name=payload.get("course_name"),
        metadata=payload["metadata"],
        transkrip=payload["transkrip"],
        ringkasan=StructuredSummary(**payload["ringkasan"]),
    )


@router.get("/adaptive-content", response_model=AdaptiveMaterialOut)
async def get_adaptive_content(lecture_id: str, is_present: bool = Query(...)):
    """
    Menyuapkan distribusi materi adaptif berbasis data presensi (FR-07 & FR-08):
    - Siswa Hadir: Menerima rekomendasi pengayaan materi lanjutan.
    - Siswa Absen: Menerima materi pendukung dasar penutup ketertinggalan.
    """
    if is_present:
        return AdaptiveMaterialOut(
            archive_id=lecture_id,
            is_present=True,
            material_type="advanced_recommendation",
            content="Materi Pengayaan: Penerapan Advanced Vector Search pgvector pada LLM Agent Pipeline.",
            links=["https://docs.boxarchive.edu/advanced-vector-search"],
        )
    else:
        return AdaptiveMaterialOut(
            archive_id=lecture_id,
            is_present=False,
            material_type="foundational_support",
            content="Materi Pendukung Dasar: Konsep Dasar Pengenalan Suara STT & Integrasi Database RAG.",
            links=["https://docs.boxarchive.edu/foundational-rag-basics"],
        )


@router.get("/{lecture_id}", response_model=ArchiveDetailOut)
async def get_lecture_archive(
    lecture_id: str,
    is_present: bool = Query(
        True,
        description="Status kehadiran mahasiswa (True=Hadir dapat transkrip penuh, False=Absen dapat ringkasan).",
    ),
):
    """
    Mengambil arsip hasil STT & Summarization 3 segmen sesuai hak akses kehadiran (FR-07 & FR-08):
    - Mahasiswa Hadir: Mendapatkan transkrip penuh + ringkasan terstruktur 3 segmen.
    - Mahasiswa Absen: Mendapatkan ringkasan terstruktur 3 segmen (tanpa transkrip mentah penuh).
    """
    archive = load_archive_by_lecture(lecture_id)
    if not archive:
        raise HTTPException(
            status_code=404,
            detail=f"Arsip untuk sesi '{lecture_id}' belum tersedia.",
        )

    return ArchiveDetailOut(
        lecture_id=archive.get("session_id", lecture_id),
        course_name=archive.get("course_name"),
        is_present=is_present,
        transkrip=archive.get("transkrip") if is_present else None,
        ringkasan=StructuredSummary(**archive["ringkasan"]),
        metadata=archive.get("metadata", {}),
    )
