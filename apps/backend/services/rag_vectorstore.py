"""
Proses 2 — Pipeline Chunking, Embedding (Gemini), & Vector Store ChromaDB
untuk RAG Chatbot Box Archive.

Dirancang modular agar mudah dimigrasi dari ChromaDB lokal ke Supabase pgvector
tanpa mengubah logika RAG Agent maupun endpoint router Chatbot.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = BACKEND_ROOT / "storage" / "chroma_db"
DEFAULT_COLLECTION_NAME = "box_archive_lectures"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def resolve_gemini_api_key() -> str:
    """
    Ambil GEMINI_API_KEY dari environment, `apps/backend/.env`,
    atau fallback ke `services/ai_pipeline/.env` saat pengujian lokal.
    """
    backend_env = BACKEND_ROOT / ".env"
    if backend_env.is_file():
        load_dotenv(backend_env, override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    pipeline_env = REPO_ROOT / "services" / "ai_pipeline" / ".env"
    if pipeline_env.is_file():
        load_dotenv(pipeline_env, override=False)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    return api_key


class LectureVectorStore:
    """
    Pengelola penyimpanan vektor transkrip & rangkuman perkuliahan berbasis ChromaDB.
    Setiap dokumen ditandai metadata `lecture_id` agar pencarian RAG terisolasi per sesi kelas.
    """

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> None:
        env_dir = os.getenv("CHROMA_PERSIST_DIR", "").strip()
        self.persist_directory = Path(
            persist_directory or env_dir or DEFAULT_CHROMA_DIR
        ).resolve()
        self.collection_name = collection_name
        self.embedding_model = (
            embedding_model
            or os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
            or DEFAULT_EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        )
        self._vectorstore: Chroma | None = None

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            api_key = resolve_gemini_api_key()
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY tidak ditemukan. Isi GEMINI_API_KEY di "
                    "apps/backend/.env atau services/ai_pipeline/.env."
                )

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                api_key=api_key,
            )
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._vectorstore

    def delete_lecture(self, lecture_id: str) -> None:
        """Hapus seluruh potongan (chunk) milik satu lecture_id."""
        store = self._get_vectorstore()
        existing = store._collection.get(where={"lecture_id": lecture_id})
        existing_ids = existing.get("ids") or []
        if existing_ids:
            store._collection.delete(ids=existing_ids)
            LOGGER.info(
                "Menghapus %d chunk lama untuk lecture_id='%s'.",
                len(existing_ids),
                lecture_id,
            )

    def has_lecture(self, lecture_id: str) -> bool:
        """Periksa apakah lecture_id sudah memiliki dokumen terindeks di ChromaDB."""
        store = self._get_vectorstore()
        existing = store._collection.get(where={"lecture_id": lecture_id}, limit=1)
        return bool(existing.get("ids"))

    def ingest_lecture(
        self,
        lecture_id: str,
        transcript: str,
        *,
        summary_text: str | None = None,
        course_name: str | None = None,
        replace_existing: bool = True,
    ) -> int:
        """
        Pecah teks transkrip (dan ringkasan opsional) menjadi chunks,
        buat embedding Gemini, lalu simpan ke ChromaDB terisolasi per `lecture_id`.
        """
        clean_lecture_id = lecture_id.strip()
        clean_transcript = transcript.strip()
        if not clean_lecture_id:
            raise ValueError("lecture_id tidak boleh kosong.")
        if not clean_transcript:
            raise ValueError("Transkrip tidak boleh kosong.")

        store = self._get_vectorstore()
        if replace_existing:
            self.delete_lecture(clean_lecture_id)

        documents: list[Document] = []
        base_meta: dict[str, Any] = {
            "lecture_id": clean_lecture_id,
            "course_name": (course_name or "").strip(),
        }

        # 1. Chunking transkrip penuh
        transcript_chunks = self.text_splitter.split_text(clean_transcript)
        for idx, chunk in enumerate(transcript_chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **base_meta,
                        "chunk_type": "transcript",
                        "chunk_index": idx,
                    },
                )
            )

        # 2. Tambahkan ringkasan terstruktur (jika tersedia) agar pencarian topik makin akurat
        if summary_text and summary_text.strip():
            summary_chunks = self.text_splitter.split_text(summary_text.strip())
            for idx, chunk in enumerate(summary_chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **base_meta,
                            "chunk_type": "summary",
                            "chunk_index": idx,
                        },
                    )
                )

        if not documents:
            return 0

        doc_ids = [
            f"{clean_lecture_id}_{doc.metadata['chunk_type']}_{doc.metadata['chunk_index']}"
            for doc in documents
        ]
        store.add_documents(documents=documents, ids=doc_ids)
        LOGGER.info(
            "Berhasil mengindeks %d chunk ke ChromaDB untuk lecture_id='%s'.",
            len(documents),
            clean_lecture_id,
        )
        return len(documents)

    def ingest_from_pipeline_json(
        self,
        json_path: Path | str,
        *,
        lecture_id: str | None = None,
        course_name: str | None = None,
        replace_existing: bool = True,
    ) -> tuple[str, int]:
        """
        Muat langsung file JSON keluaran `services/ai_pipeline/main_pipeline.py`
        dan indeks transkrip + ringkasan 3 segmen ke ChromaDB.
        """
        path = Path(json_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File JSON arsip tidak ditemukan: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        resolved_lecture_id = (
            lecture_id or str(payload.get("session_id") or path.stem)
        ).strip()
        transcript = str(payload.get("transkrip", "")).strip()

        ringkasan = payload.get("ringkasan") or {}
        summary_parts: list[str] = []
        if isinstance(ringkasan, dict):
            poin = ringkasan.get("poin_utama_materi") or []
            qa = ringkasan.get("sesi_tanya_jawab") or []
            tugas = ringkasan.get("tugas") or []
            if poin:
                summary_parts.append(
                    "Poin Utama Materi:\n- " + "\n- ".join(str(x) for x in poin)
                )
            if qa:
                summary_parts.append(
                    "Sesi Tanya Jawab:\n- " + "\n- ".join(str(x) for x in qa)
                )
            if tugas:
                summary_parts.append("Tugas:\n- " + "\n- ".join(str(x) for x in tugas))

        summary_text = "\n\n".join(summary_parts) if summary_parts else None
        count = self.ingest_lecture(
            lecture_id=resolved_lecture_id,
            transcript=transcript,
            summary_text=summary_text,
            course_name=course_name or payload.get("course_name"),
            replace_existing=replace_existing,
        )
        return resolved_lecture_id, count

    def retrieve_context(
        self,
        lecture_id: str,
        query: str,
        *,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """
        Cari potongan teks paling relevan dari ChromaDB HANYA untuk `lecture_id` terkait.
        Mengembalikan daftar tuple `(Document, distance_score)` (semakin kecil jarak = semakin relevan).
        """
        clean_lecture_id = lecture_id.strip()
        clean_query = query.strip()
        if not clean_lecture_id or not clean_query:
            return []

        store = self._get_vectorstore()
        if not self.has_lecture(clean_lecture_id):
            return []

        return store.similarity_search_with_score(
            query=clean_query,
            k=k,
            filter={"lecture_id": clean_lecture_id},
        )


vector_store = LectureVectorStore()
