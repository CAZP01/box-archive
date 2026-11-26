"""
Proses 4 — Endpoint REST API khusus RAG Chatbot Box Archive:
- POST   /api/v1/chatbot/ingest                           : Indeks transkrip sesi kelas ke ChromaDB
- POST   /api/v1/chatbot/query                            : Tanya jawab RAG + Guardrail anti-halusinasi
- GET    /api/v1/chatbot/history/{lecture_id}/{student_id}: Ambil riwayat chat mahasiswa
- DELETE /api/v1/chatbot/history/{lecture_id}/{student_id}: Reset riwayat chat mahasiswa
"""

from fastapi import APIRouter, HTTPException

try:
    from schema.chatbot import (
        ChatHistoryResponse,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        IngestTranscriptRequest,
        IngestTranscriptResponse,
    )
    from services.langchain_agent import rag_agent
except ImportError:
    from app.schemas.chatbot import (
        ChatHistoryResponse,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        IngestTranscriptRequest,
        IngestTranscriptResponse,
    )
    from app.services.langchain_agent import rag_agent

router = APIRouter(prefix="/api/v1/chatbot", tags=["RAG Chatbot Agent"])


@router.post("/ingest", response_model=IngestTranscriptResponse)
async def ingest_lecture_transcript(payload: IngestTranscriptRequest):
    """
    Memecah (chunking) dan menyimpan transkrip/rangkuman perkuliahan hasil Proses 3.0
    ke Vector Store ChromaDB agar dapat ditanyakan melalui RAG Chatbot.
    """
    try:
        chunks_indexed = rag_agent.vector_store.ingest_lecture(
            lecture_id=payload.lecture_id,
            transcript=payload.transcript,
            summary_text=payload.summary_text,
            course_name=payload.course_name,
            replace_existing=payload.replace_existing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengindeks transkrip ke ChromaDB: {exc}",
        ) from exc

    return IngestTranscriptResponse(
        status="success",
        lecture_id=payload.lecture_id,
        chunks_indexed=chunks_indexed,
        vector_store="chromadb",
        embedding_model=rag_agent.vector_store.embedding_model,
    )


@router.post("/query", response_model=ChatResponse)
async def query_chatbot(payload: ChatRequest):
    """
    Endpoint RAG Chatbot dengan Guardrail anti-halusinasi (NFR Reliability)
    dan pencatatan riwayat percakapan per mahasiswa.
    """
    try:
        result = await rag_agent.answer_question(
            question=payload.question,
            lecture_id=payload.lecture_id,
            student_id=payload.student_id,
            save_history=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses pertanyaan chatbot: {exc}",
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        source_context=result["source_context"],
        is_out_of_context=result["is_out_of_context"],
        lecture_id=payload.lecture_id,
        student_id=payload.student_id,
        retrieved_chunks=result["retrieved_chunks"],
        response_time_seconds=result["response_time_seconds"],
    )


@router.get("/history/{lecture_id}/{student_id}", response_model=ChatHistoryResponse)
async def get_chat_history(lecture_id: str, student_id: str):
    """Ambil seluruh riwayat percakapan mahasiswa pada sesi perkuliahan tertentu."""
    raw_messages = rag_agent.history_store.get_messages(lecture_id, student_id)
    messages = [ChatMessage(**msg) for msg in raw_messages]
    return ChatHistoryResponse(
        lecture_id=lecture_id,
        student_id=student_id,
        total_messages=len(messages),
        messages=messages,
    )


@router.delete("/history/{lecture_id}/{student_id}")
async def clear_chat_history(lecture_id: str, student_id: str):
    """Hapus riwayat percakapan mahasiswa pada sesi perkuliahan tertentu."""
    rag_agent.history_store.clear_messages(lecture_id, student_id)
    return {
        "status": "cleared",
        "lecture_id": lecture_id,
        "student_id": student_id,
    }
