from fastapi import APIRouter, Query, HTTPException
from app.schemas.archive import AdaptiveMaterialOut

router = APIRouter(prefix="/api/v1/archives", tags=["Archives & Adaptive Distribution"])

@router.get("/adaptive-content", response_model=AdaptiveMaterialOut)
async def get_adaptive_content(lecture_id: str, is_present: bool = Query(...)):
    """
    Menyuapkan distribusi materi adaptif berbasis data presensi[cite: 1]:
    - Siswa Hadir: Menerima rekomendasi pengayaan materi lanjutan[cite: 1].
    - Siswa Absen: Menerima materi pendukung dasar penutup ketertinggalan[cite: 1].
    """
    if is_present:
        return AdaptiveMaterialOut(
            archive_id=lecture_id,
            is_present=True,
            material_type="advanced_recommendation",
            content="Materi Pengayaan: Penerapan Advanced Vector Search pgvector pada LLM Agent Pipeline.",
            links=["https://docs.boxarchive.edu/advanced-vector-search"]
        )
    else:
        return AdaptiveMaterialOut(
            archive_id=lecture_id,
            is_present=False,
            material_type="foundational_support",
            content="Materi Pendukung Dasar: Konsep Dasar Pengenalan Suara STT & Integrasi Database RAG.",
            links=["https://docs.boxarchive.edu/foundational-rag-basics"]
        )