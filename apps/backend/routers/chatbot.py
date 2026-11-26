from fastapi import APIRouter, HTTPException
from app.schemas.chatbot import ChatRequest, ChatResponse
from app.services.langchain_agent import rag_agent

router = APIRouter(prefix="/api/v1/chatbot", tags=["RAG Chatbot Agent"])

async def get_transcript_by_lecture(lecture_id: str) -> str:
    return """
    Pada perkuliahan hari ini kita membahas tentang arsitektur IoT berbasis MQTT yang dihubungkan ke FastAPI. 
    Dosen menjelaskan bahwa sensor ESP32 mempublikasikan data status ke broker, yang kemudian diproses oleh backend. 
    Tugas minggu depan adalah mengimplementasikan antrean pesan MQTT secara lokal.
    """

@router.post("/query", response_model=ChatResponse)
async def query_chatbot(payload: ChatRequest):
    """
    Endpoint RAG Chatbot dengan Guardrail bebas halusinasi[cite: 1].
    """
    transcript_context = await get_transcript_by_lecture(payload.lecture_id)
    result = await rag_agent.answer_question(payload.question, transcript_context)
    
    return ChatResponse(
        answer=result["answer"],
        source_context=transcript_context if not result["is_out_of_context"] else None,
        is_out_of_context=result["is_out_of_context"]
    )