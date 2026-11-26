from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ArchiveSummary(BaseModel):
    id: str
    lecture_id: str
    course_name: str
    date: datetime
    main_points: List[str]
    qa_session: List[dict]
    assignments: List[str]


class AdaptiveMaterialOut(BaseModel):
    archive_id: str
    is_present: bool
    material_type: str
    content: str
    links: List[str]


class StructuredSummary(BaseModel):
    """Skema ringkasan terstruktur 3 segmen sesuai FR-04 dan NFR-07."""

    poin_utama_materi: List[str] = Field(default_factory=list)
    sesi_tanya_jawab: List[str] = Field(default_factory=list)
    tugas: List[str] = Field(default_factory=list)


class ArchiveProcessResponse(BaseModel):
    """Respons setelah Proses 3.0 (STT + Summarization + Hapus Audio Mentah) selesai."""

    status: str
    lecture_id: str
    course_name: Optional[str] = None
    metadata: dict[str, Any]
    transkrip: str
    ringkasan: StructuredSummary


class ArchiveDetailOut(BaseModel):
    """Respons akses arsip adaptif per mahasiswa (FR-07 Hadir vs FR-08 Absen)."""

    lecture_id: str
    course_name: Optional[str] = None
    is_present: bool
    transkrip: Optional[str] = None
    ringkasan: StructuredSummary
    metadata: dict[str, Any]
