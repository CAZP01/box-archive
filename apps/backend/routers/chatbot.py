from fastapi import APIRouter

from schema.chatbot import ChatRequest, ChatResponse
from services.ai_pipeline_bridge import load_transcript_text
from services.langchain_agent import rag_agent

router = APIRouter(prefix="/api/v1/chatbot", tags=["RAG Chatbot Agent"])

_FALLBACK_SAMPLE_TRANSCRIPT = """
Pada perkuliahan hari ini kita membahas tentang arsitektur IoT berbasis MQTT yang dihubungkan ke FastAPI.
Dosen menjelaskan bahwa sensor ESP32 mempublikasikan data status ke broker, yang kemudian diproses oleh backend.
Tugas minggu depan adalah mengimplementasikan antrean pesan MQTT secara lokal.
"""


async def get_transcript_by_lecture(lecture_id: str) -> str:
    """Ambil transkrip asli hasil Proses 3.0 (ai_pipeline) jika tersedia; jika belum, pakai sampel."""
    saved_transcript = load_transcript_text(lecture_id)
    if saved_transcript:
        return saved_transcript
    return _FALLBACK_SAMPLE_TRANSCRIPT


@router.post("/query", response_model=ChatResponse)
async def query_chatbot(payload: ChatRequest):
    """
    Endpoint RAG Chatbot dengan Guardrail bebas halusinasi (NFR Reliability).
    """
    transcript_context = await get_transcript_by_lecture(payload.lecture_id)
    result = await rag_agent.answer_question(payload.question, transcript_context)

    return ChatResponse(
        answer=result["answer"],
        source_context=transcript_context if not result["is_out_of_context"] else None,
        is_out_of_context=result["is_out_of_context"],
    )
