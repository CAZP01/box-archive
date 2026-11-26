"""
Box Archive — AI Pipeline (Proses 3.0: Speech-to-Text & Summarization).

Penanggung Jawab: M. Alan Daulay (23106050016)
"""

from .main_pipeline import DEFAULT_OUTPUT_PATH, PROCESSING_TARGET_SECONDS, run_pipeline
from .stt_processor import (
    TranscriptionResult,
    TranscriptSegment,
    transcribe_audio,
    transcribe_audio_detailed,
    validate_audio_file,
)
from .summarizer import (
    SummaryOutput,
    fallback_summarize_transcript,
    summarize_transcript,
)

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "PROCESSING_TARGET_SECONDS",
    "SummaryOutput",
    "TranscriptSegment",
    "TranscriptionResult",
    "fallback_summarize_transcript",
    "run_pipeline",
    "summarize_transcript",
    "transcribe_audio",
    "transcribe_audio_detailed",
    "validate_audio_file",
]
