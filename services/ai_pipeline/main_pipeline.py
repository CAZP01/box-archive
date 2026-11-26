import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

try:
    from .stt_processor import transcribe_audio_detailed, validate_audio_file
    from .summarizer import summarize_transcript
except ImportError:
    from stt_processor import transcribe_audio_detailed, validate_audio_file
    from summarizer import summarize_transcript

load_dotenv(Path(__file__).with_name(".env"))

LOGGER = logging.getLogger(__name__)
PROCESSING_TARGET_SECONDS = 30 * 60
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("output_hasil_kelas.json")


def run_pipeline(
    audio_path: Path | str,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    session_id: str | None = None,
    delete_audio: bool = True,
    allow_fallback_on_error: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Jalankan Proses 3.0 (Rekam dan Proses Arsip: STT & Ringkasan 3 Segmen).

    Alur Sesuai Proposal Capstone Box Archive (FR-04, FR-05, NFR-05, NFR-07, NFR-08):
    1. Validasi file rekaman audio kelas.
    2. Konversi suara menjadi teks Bahasa Indonesia (Faster-Whisper).
    3. Ekstraksi rangkuman terstruktur menjadi 3 segmen:
       - Poin utama materi (`poin_utama_materi`)
       - Sesi tanya jawab (`sesi_tanya_jawab`)
       - Tugas (`tugas`)
    4. Simpan transkrip dan rangkuman secara atomik ke file output (D3 Arsip).
    5. Hapus file audio mentah HANYA SETELAH transkrip dan rangkuman berhasil tersimpan
       (FR-05 & NFR-05). Jika terjadi kegagalan di tengah proses, file audio mentah
       dipertahankan agar dapat diproses ulang (retry).
    """
    input_path = validate_audio_file(audio_path)
    result_path = Path(output_path).expanduser().resolve()

    if result_path == input_path:
        raise ValueError("Path output JSON tidak boleh sama dengan file audio input.")

    target_seconds = int(
        os.getenv("MAX_PROCESSING_SECONDS", str(PROCESSING_TARGET_SECONDS))
    )
    if target_seconds <= 0:
        raise ValueError("MAX_PROCESSING_SECONDS harus berupa bilangan bulat positif.")

    pipeline_started_at = time.perf_counter()
    LOGGER.info("Memulai pemrosesan arsip audio kelas: %s", input_path)

    # Tahap 1: Speech-to-Text (Faster-Whisper)
    stt_result = transcribe_audio_detailed(
        input_path,
        show_progress=show_progress,
    )
    LOGGER.info(
        "Transkripsi selesai dalam %.2f detik (%d segmen, durasi audio %.2f detik). "
        "Memulai summarization...",
        stt_result.processing_seconds,
        len(stt_result.segments),
        stt_result.audio_duration_seconds,
    )

    # Tahap 2: Summarization 3 Segmen (LangChain + Gemini 2.5 Flash / Fallback Lokal)
    summary_started_at = time.perf_counter()
    summary = summarize_transcript(
        stt_result.text,
        allow_fallback_on_error=allow_fallback_on_error,
    )
    summary_seconds = time.perf_counter() - summary_started_at

    total_processing_seconds = time.perf_counter() - pipeline_started_at
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model_name = (
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    )
    summarization_model = (
        gemini_model_name if has_gemini_key else "fallback-lokal-aturan-kata-kunci"
    )

    result_payload = {
        "session_id": session_id or input_path.stem,
        "metadata": {
            "timestamp_selesai_utc": datetime.now(timezone.utc).isoformat(),
            "durasi_audio_detik": stt_result.audio_duration_seconds,
            "waktu_stt_detik": stt_result.processing_seconds,
            "waktu_summarization_detik": round(summary_seconds, 2),
            "waktu_pemrosesan_detik": round(total_processing_seconds, 2),
            "memenuhi_target_30_menit": total_processing_seconds <= target_seconds,
            "target_detik": target_seconds,
            "bahasa": stt_result.language,
            "probabilitas_bahasa": stt_result.language_probability,
            "jumlah_segmen_transkrip": len(stt_result.segments),
            "status_audio_mentah": (
                "dihapus_setelah_tersimpan"
                if delete_audio
                else "dipertahankan_mode_uji"
            ),
            "model": {
                "speech_to_text": f"faster-whisper/{stt_result.model_size}",
                "summarization": summarization_model,
            },
        },
        "transkrip": stt_result.text,
        "segmen_transkrip": [
            seg.model_dump(mode="json") for seg in stt_result.segments
        ],
        "ringkasan": summary.model_dump(mode="json"),
    }

    # Tahap 3: Simpan ke D3 Arsip secara atomik (via file sementara terlebih dahulu)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = result_path.with_suffix(result_path.suffix + ".tmp")
    try:
        with temp_output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result_payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temp_output_path.replace(result_path)
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink(missing_ok=True)

    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise IOError(f"Gagal memverifikasi penyimpanan file arsip: {result_path}")

    # Tahap 4: Integritas Penghapusan Audio Mentah (FR-05 & NFR-05)
    # Audio mentah HANYA dihapus setelah transkrip & rangkuman dipastikan tersimpan.
    if delete_audio:
        os.remove(input_path)
        LOGGER.info(
            "Audio mentah berhasil dihapus sesuai kebijakan privasi (FR-05/NFR-05): %s",
            input_path,
        )
    else:
        LOGGER.info(
            "Mode uji (--keep-audio): File audio mentah dipertahankan di %s",
            input_path,
        )

    if total_processing_seconds <= target_seconds:
        LOGGER.info(
            "Pemrosesan arsip selesai dalam %.2f detik (memenuhi target <= %d detik).",
            total_processing_seconds,
            target_seconds,
        )
    else:
        LOGGER.warning(
            "Pemrosesan memakan %.2f detik, melebihi target %d detik (30 menit).",
            total_processing_seconds,
            target_seconds,
        )

    LOGGER.info("Arsip transkrip & rangkuman tersimpan di: %s", result_path)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Box Archive AI Pipeline (Proses 3.0): Transkripsikan audio kelas "
            "berbahasa Indonesia dan rangkum ke dalam 3 segmen terstruktur."
        )
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path file rekaman audio kelas (.mp3, .wav, .m4a, .webm, dll.).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path output JSON arsip kelas (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="ID sesi perkuliahan (opsional, default menggunakan nama file audio).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help=(
            "Pertahankan file audio mentah setelah proses selesai (khusus mode "
            "testing/development). Secara default, audio dihapus sesuai FR-05/NFR-05."
        ),
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "Jika pemanggilan Gemini API gagal setelah retry, gunakan ekstraksi "
            "kata kunci lokal sebagai cadangan."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Sembunyikan pencetakan log segmen transkrip per baris di terminal.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        run_pipeline(
            args.audio_path,
            args.output,
            session_id=args.session_id,
            delete_audio=not args.keep_audio,
            allow_fallback_on_error=args.allow_fallback,
            show_progress=not args.quiet,
        )
    except Exception:
        LOGGER.exception(
            "Pipeline gagal (NFR-05/NFR-08). File audio mentah dipertahankan "
            "untuk pemrosesan ulang. Periksa input, konfigurasi, dan log error."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
