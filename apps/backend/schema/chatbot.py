from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestTranscriptRequest(BaseModel):
    """Input untuk memasukkan transkrip/rangkuman satu sesi kelas ke Vector Store (ChromaDB)."""

    lecture_id: str = Field(..., min_length=1, description="ID unik sesi perkuliahan.")
    transcript: str = Field(
        ..., min_length=1, description="Teks transkrip utuh hasil Speech-to-Text (Proses 3.0)."
    )
    summary_text: Optional[str] = Field(
        default=None,
        description="Teks rangkuman terstruktur 3 segmen (opsional, untuk memperkaya konteks RAG).",
    )
    course_name: Optional[str] = Field(
        default=None, description="Nama mata kuliah (metadata pencarian)."
    )
    replace_existing: bool = Field(
        default=True,
        description="Jika True, hapus chunk lama untuk lecture_id yang sama sebelum menyimpan ulang.",
    )


class IngestTranscriptResponse(BaseModel):
    """Respons setelah transkrip berhasil dipecah (chunking) dan disimpan ke Vector Store."""

    status: str
    lecture_id: str
    chunks_indexed: int
    vector_store: str = "chromadb"
    embedding_model: str


class ChatRequest(BaseModel):
    """Permintaan pertanyaan mahasiswa ke RAG Chatbot."""

    lecture_id: str = Field(..., min_length=1, description="ID sesi perkuliahan yang ditanyakan.")
    question: str = Field(..., min_length=1, description="Pertanyaan mahasiswa.")
    student_id: str = Field(
        default="anonymous",
        description="ID mahasiswa untuk memisahkan riwayat percakapan (Entitas Riwayat Chatbot).",
    )


class ChatMessage(BaseModel):
    """Satu entri pesan dalam riwayat percakapan mahasiswa."""

    role: Literal["student", "assistant"]
    content: str
    timestamp_utc: str
    is_out_of_context: bool = False


class ChatResponse(BaseModel):
    """Respons RAG Chatbot dengan Guardrail anti-halusinasi dan metrik kinerja (NFR-10)."""

    answer: str
    source_context: Optional[str] = None
    is_out_of_context: bool = False
    lecture_id: Optional[str] = None
    student_id: Optional[str] = None
    retrieved_chunks: int = 0
    response_time_seconds: float = 0.0


class ChatHistoryResponse(BaseModel):
    """Riwayat percakapan mahasiswa untuk satu sesi perkuliahan."""

    lecture_id: str
    student_id: str
    total_messages: int
    messages: list[ChatMessage] = Field(default_factory=list)
