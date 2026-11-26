import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydantic import BaseModel, ConfigDict, Field

load_dotenv(Path(__file__).with_name(".env"))

LOGGER = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".webm",
    ".ogg",
    ".flac",
    ".aac",
    ".mp4",
}

DEFAULT_INITIAL_PROMPT = (
    "Berikut adalah rekaman transkrip perkuliahan, penjelasan materi oleh dosen, "
    "sesi tanya jawab, dan penugasan akademik di kelas dalam Bahasa Indonesia baku."
)

# Regex untuk membersihkan karakter non-Latin anomali (seperti hallucination CJK pada Whisper)
_NON_LATIN_HALLUCINATION_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+"
)
_MULTISPACE_RE = re.compile(r"\s+")

_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


class TranscriptSegment(BaseModel):
    """Satu segmen kalimat transkrip beserta rentang waktu (detik)."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(description="Waktu mulai segmen dalam detik.")
    end: float = Field(description="Waktu selesai segmen dalam detik.")
    text: str = Field(description="Teks hasil transkripsi pada segmen ini.")


class TranscriptionResult(BaseModel):
    """Hasil lengkap proses Speech-to-Text beserta metadata evaluasi (NFR-07)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Transkrip utuh hasil penggabungan seluruh segmen.")
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Daftar segmen transkrip ber-timestamp.",
    )
    language: str = Field(default="id", description="Kode bahasa yang terdeteksi.")
    language_probability: float = Field(
        default=1.0,
        description="Probabilitas deteksi bahasa (0.0 - 1.0).",
    )
    audio_duration_seconds: float = Field(
        default=0.0,
        description="Durasi total rekaman audio dalam detik.",
    )
    processing_seconds: float = Field(
        default=0.0,
        description="Waktu pemrosesan STT dalam detik.",
    )
    model_size: str = Field(
        default="base",
        description="Ukuran model Faster-Whisper yang digunakan.",
    )


def _terminal_safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def _clean_segment_text(raw_text: str) -> str:
    """Bersihkan karakter hallucination non-Latin dan rapikan spasi."""
    cleaned = _NON_LATIN_HALLUCINATION_RE.sub(" ", raw_text)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _get_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    """Ambil instance WhisperModel dari cache agar tidak reload berulang kali."""
    cache_key = (model_size, device, compute_type)
    if cache_key not in _MODEL_CACHE:
        LOGGER.info(
            "Memuat model Faster-Whisper (size=%s, device=%s, compute_type=%s)...",
            model_size,
            device,
            compute_type,
        )
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _MODEL_CACHE[cache_key]


def validate_audio_file(audio_path: Union[str, os.PathLike]) -> Path:
    """Validasi keberadaan, ekstensi, dan ukuran file audio sebelum diproses."""
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File audio tidak ditemukan: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File audio kosong (0 byte): {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        LOGGER.warning(
            "Ekstensi file %s tidak termasuk daftar standar (%s), tetap mencoba memproses.",
            path.suffix,
            ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS)),
        )
    return path


def transcribe_audio_detailed(
    audio_path: Union[str, os.PathLike],
    *,
    show_progress: bool = True,
) -> TranscriptionResult:
    """
    Transkripsikan rekaman audio kelas berbahasa Indonesia menggunakan Faster-Whisper.

    Catatan Arsitektur (FR-05 & NFR-05):
    Fungsi ini TIDAK menghapus file audio mentah secara langsung. Penghapusan file audio
    wajib dilakukan oleh orkestrator pipeline HANYA SETELAH transkrip dan rangkuman
    berhasil disimpan, sehingga jika tahap summarization/penyimpanan gagal, audio
    tetap tersedia untuk diproses ulang (retry).
    """
    validated_path = validate_audio_file(audio_path)

    model_size = os.getenv("WHISPER_MODEL_SIZE", "base").strip() or "base"
    device = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    max_processing_seconds = int(os.getenv("MAX_PROCESSING_SECONDS", "1800"))

    if max_processing_seconds <= 0:
        raise ValueError("MAX_PROCESSING_SECONDS harus berupa bilangan bulat positif.")
    if beam_size <= 0:
        raise ValueError("WHISPER_BEAM_SIZE harus berupa bilangan bulat positif.")

    model = _get_whisper_model(model_size, device, compute_type)
    started_at = time.perf_counter()

    segments_iter, info = model.transcribe(
        str(validated_path),
        language="id",
        initial_prompt=DEFAULT_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        beam_size=beam_size,
        condition_on_previous_text=False,
    )

    transcript_texts: list[str] = []
    structured_segments: list[TranscriptSegment] = []

    for segment in segments_iter:
        cleaned_text = _clean_segment_text(segment.text)
        if not cleaned_text:
            continue

        transcript_texts.append(cleaned_text)
        structured_segments.append(
            TranscriptSegment(
                start=round(float(segment.start), 2),
                end=round(float(segment.end), 2),
                text=cleaned_text,
            )
        )
        if show_progress:
            log_line = f"[{segment.start:.2f} -> {segment.end:.2f}] {cleaned_text}"
            print(_terminal_safe_text(log_line), flush=True)

    full_transcript = " ".join(transcript_texts).strip()
    elapsed_seconds = time.perf_counter() - started_at

    if elapsed_seconds > max_processing_seconds:
        warnings.warn(
            f"Transkripsi memakan waktu {elapsed_seconds:.2f} detik, melebihi batas "
            f"{max_processing_seconds} detik (NFR Performance).",
            RuntimeWarning,
            stacklevel=2,
        )

    return TranscriptionResult(
        text=full_transcript,
        segments=structured_segments,
        language=getattr(info, "language", "id") or "id",
        language_probability=round(
            float(getattr(info, "language_probability", 1.0) or 1.0), 4
        ),
        audio_duration_seconds=round(
            float(getattr(info, "duration", 0.0) or 0.0), 2
        ),
        processing_seconds=round(elapsed_seconds, 2),
        model_size=model_size,
    )


def transcribe_audio(audio_path: Union[str, os.PathLike]) -> str:
    """Fungsi kompatibilitas yang mengembalikan teks transkrip utuh sebagai string."""
    result = transcribe_audio_detailed(audio_path)
    return result.text
