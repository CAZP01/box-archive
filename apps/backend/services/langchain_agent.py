from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings


class BoxArchiveRAGAgent:
    def __init__(self):
        self._llm: ChatGoogleGenerativeAI | None = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=settings.GEMINI_API_KEY,
                temperature=0.2,
            )
        return self._llm

    async def generate_summary(self, transcript_text: str) -> dict:
        """Memproses transkrip Speech-to-Text menjadi ringkasan terstruktur."""
        prompt = f"""
        Kamu adalah asisten pengarsip kelas cerdas Box Archive.
        Ekstrak transkrip perkuliahan berikut ke dalam format ringkasan terstruktur:
        1. Poin Utama Materi
        2. Sesi Tanya Jawab
        3. Tugas / Informasi Perkuliahan

        Transkrip:
        {transcript_text}
        """
        response = await self._get_llm().ainvoke(prompt)
        return {"summary_text": response.content}

    async def answer_question(self, question: str, transcript_context: str) -> dict:
        """
        Memberikan jawaban chatbot berbasis RAG dengan Guardrail kontekstual.
        Jika pertanyaan di luar transkrip, berikan penolakan baku.
        """
        guardrail_prompt = f"""
        Kamu adalah chatbot asisten perkuliahan Box Archive.
        Jawab pertanyaan siswa HANYA berdasarkan konteks transkrip kelas hari ini berikut ini.
        Jika jawaban TIDAK ADA di dalam konteks transkrip, kamu WAJIB menjawab persis:
        "Maaf, informasi ini tidak terdapat dalam rekaman kelas hari ini."

        Konteks Transkrip:
        {transcript_context}

        Pertanyaan: {question}
        """
        response = await self._get_llm().ainvoke(guardrail_prompt)
        answer_text = str(response.content).strip()

        is_out_of_context = (
            "Maaf, informasi ini tidak terdapat dalam rekaman kelas hari ini"
            in answer_text
        )
        return {
            "answer": answer_text,
            "is_out_of_context": is_out_of_context,
        }


rag_agent = BoxArchiveRAGAgent()
