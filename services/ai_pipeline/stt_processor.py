import os
import sys
import time
import warnings
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv(Path(__file__).with_name(".env"))


def _terminal_safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def transcribe_audio(audio_path: Union[str, os.PathLike]) -> str:
    """Transcribe an Indonesian audio file and delete it after success."""
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    max_processing_seconds = int(os.getenv("MAX_PROCESSING_SECONDS", "1800"))
    if max_processing_seconds <= 0:
        raise ValueError("MAX_PROCESSING_SECONDS must be a positive integer.")

    started_at = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_path,
        language="id",
        vad_filter=True,
        beam_size=5,
    )

    transcript_segments = []
    for segment in segments:
        text = segment.text.strip()
        transcript_segments.append(text)
        log_line = f"[{segment.start:.2f} -> {segment.end:.2f}] {text}"
        print(_terminal_safe_text(log_line), flush=True)

    transcript = " ".join(transcript_segments).strip()
    elapsed_seconds = time.perf_counter() - started_at
    os.remove(audio_path)

    if elapsed_seconds > max_processing_seconds:
        warnings.warn(
            f"Transcription took {elapsed_seconds:.2f} seconds, exceeding the "
            f"{max_processing_seconds}-second limit.",
            RuntimeWarning,
            stacklevel=2,
        )

    return transcript
