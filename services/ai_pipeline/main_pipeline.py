import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from stt_processor import transcribe_audio
from summarizer import summarize_transcript

load_dotenv(Path(__file__).with_name(".env"))

LOGGER = logging.getLogger(__name__)
PROCESSING_TARGET_SECONDS = 30 * 60
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("output_hasil_kelas.json")


def run_pipeline(audio_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Transcribe, summarize, and save one class recording."""
    input_path = Path(audio_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"File audio tidak ditemukan: {input_path}")

    result_path = Path(output_path).expanduser().resolve()
    if result_path == input_path:
        raise ValueError("Path output JSON tidak boleh sama dengan file audio input.")

    started_at = time.perf_counter()
    LOGGER.info("Memulai pemrosesan audio: %s", input_path)

    transcript = transcribe_audio(str(input_path))
    LOGGER.info("Transkripsi selesai; memulai summarization.")
    summary = summarize_transcript(transcript)

    processing_seconds = time.perf_counter() - started_at
    whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    summarization_model = (
        "gemini-2.5-flash"
        if os.getenv("GEMINI_API_KEY", "").strip()
        else "fallback-lokal-aturan-kata-kunci"
    )
    result = {
        "metadata": {
            "waktu_pemrosesan_detik": round(processing_seconds, 2),
            "memenuhi_target_30_menit": (
                processing_seconds < PROCESSING_TARGET_SECONDS
            ),
            "target_detik": PROCESSING_TARGET_SECONDS,
            "model": {
                "speech_to_text": f"faster-whisper/{whisper_model_size}",
                "summarization": summarization_model,
            },
        },
        "transkrip": transcript,
        "ringkasan": summary.model_dump(mode="json"),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    if processing_seconds < PROCESSING_TARGET_SECONDS:
        LOGGER.info("Pemrosesan selesai dalam %.2f detik.", processing_seconds)
    else:
        LOGGER.warning(
            "Pemrosesan memakan %.2f detik, melebihi target 30 menit.",
            processing_seconds,
        )
    LOGGER.info("Hasil tersimpan di: %s", result_path)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transkripsikan dan rangkum rekaman kelas."
    )
    parser.add_argument("audio_path", type=Path, help="Path file audio input.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path output JSON (default: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        run_pipeline(args.audio_path, args.output)
    except Exception:
        LOGGER.exception("Pipeline gagal. Periksa input, konfigurasi, dan log error.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
