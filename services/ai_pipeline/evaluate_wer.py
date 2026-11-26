import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

try:
    from .stt_processor import transcribe_audio_detailed
except ImportError:
    from stt_processor import transcribe_audio_detailed

LOGGER = logging.getLogger(__name__)

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")
REQUIRED_SUMMARY_SEGMENTS = ("poin_utama_materi", "sesi_tanya_jawab", "tugas")


class AlignmentCounts(BaseModel):
    """Rincian operasi edit Levenshtein antara teks referensi dan hipotesis."""

    model_config = ConfigDict(extra="forbid")

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0
    hypothesis_length: int = 0
    error_rate_percent: float = 0.0


class EvaluationReport(BaseModel):
    """Laporan evaluasi akurasi STT & kelengkapan rangkuman (NFR-07 & NFR Performance)."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    model_name: str
    target_wer_percent: float
    wer_percent: float
    cer_percent: float
    memenuhi_target_wer: bool
    word_alignment: AlignmentCounts
    char_alignment: AlignmentCounts
    audio_duration_seconds: float | None = None
    processing_seconds: float | None = None
    real_time_factor: float | None = None
    estimasi_kuliah_90_menit_detik: float | None = None
    memenuhi_target_30_menit_estimasi: bool | None = None
    kelengkapan_3_segmen_ringkasan: bool | None = None
    catatan_pengujian: str = Field(
        default="Evaluasi standar WER/CER menggunakan normalisasi huruf kecil tanpa tanda baca."
    )


def normalize_text(text: str) -> str:
    """Normalisasi teks Bahasa Indonesia untuk evaluasi ASR (lowercase, hapus tanda baca)."""
    lowered = text.lower()
    without_punct = _PUNCTUATION_RE.sub(" ", lowered)
    return _MULTISPACE_RE.sub(" ", without_punct).strip()


def _levenshtein_counts(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> AlignmentCounts:
    """
    Hitung jarak edit Levenshtein lengkap (Substitutions, Deletions, Insertions)
    dengan dynamic programming.
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)

    if n == 0:
        err_rate = 0.0 if m == 0 else 100.0
        return AlignmentCounts(
            substitutions=0,
            deletions=0,
            insertions=m,
            reference_length=0,
            hypothesis_length=m,
            error_rate_percent=round(err_rate, 2),
        )

    # dp[i][j] = (total_cost, substitutions, deletions, insertions)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        r_tok = ref_tokens[i - 1]
        for j in range(1, m + 1):
            h_tok = hyp_tokens[j - 1]
            if r_tok == h_tok:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost, sub_s, sub_d, sub_i = dp[i - 1][j - 1]
                del_cost, del_s, del_d, del_i = dp[i - 1][j]
                ins_cost, ins_s, ins_d, ins_i = dp[i][j - 1]

                cand_sub = (sub_cost + 1, sub_s + 1, sub_d, sub_i)
                cand_del = (del_cost + 1, del_s, del_d + 1, del_i)
                cand_ins = (ins_cost + 1, ins_s, ins_d, ins_i + 1)

                dp[i][j] = min(cand_sub, cand_del, cand_ins, key=lambda x: x[0])

    total_errors, subs, dels, ins = dp[n][m]
    error_rate = (total_errors / n) * 100.0
    return AlignmentCounts(
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_length=n,
        hypothesis_length=m,
        error_rate_percent=round(error_rate, 2),
    )


def compute_wer_cer(reference_text: str, hypothesis_text: str) -> tuple[AlignmentCounts, AlignmentCounts]:
    """Hitung Word Error Rate (WER) dan Character Error Rate (CER)."""
    norm_ref = normalize_text(reference_text)
    norm_hyp = normalize_text(hypothesis_text)

    ref_words = norm_ref.split() if norm_ref else []
    hyp_words = norm_hyp.split() if norm_hyp else []
    word_stats = _levenshtein_counts(ref_words, hyp_words)

    ref_chars = list(norm_ref)
    hyp_chars = list(norm_hyp)
    char_stats = _levenshtein_counts(ref_chars, hyp_chars)

    return word_stats, char_stats


def _load_hypothesis_file(path: Path) -> tuple[str, bool | None, str]:
    """Muat teks hipotesis dari file .txt atau file .json hasil main_pipeline.py."""
    content = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        payload = json.loads(content)
        transcript = str(payload.get("transkrip", ""))
        summary = payload.get("ringkasan")
        has_three_segments = (
            isinstance(summary, dict)
            and all(seg in summary and isinstance(summary[seg], list) for seg in REQUIRED_SUMMARY_SEGMENTS)
        )
        model_name = (
            payload.get("metadata", {})
            .get("model", {})
            .get("speech_to_text", "faster-whisper/unknown")
        )
        return transcript, has_three_segments, model_name
    return content, None, "file-hipotesis-teks"


def evaluate_transcript(
    reference_text: str,
    hypothesis_text: str,
    *,
    model_name: str = "faster-whisper/base",
    target_wer_percent: float = 20.0,
    audio_duration_seconds: float | None = None,
    processing_seconds: float | None = None,
    has_three_segments: bool | None = None,
) -> EvaluationReport:
    """Bangun laporan evaluasi terstruktur untuk NFR-07 dan NFR Performance."""
    word_stats, char_stats = compute_wer_cer(reference_text, hypothesis_text)

    rtf: float | None = None
    est_90m_seconds: float | None = None
    meets_30m_est: bool | None = None

    if audio_duration_seconds and audio_duration_seconds > 0 and processing_seconds is not None:
        rtf = round(processing_seconds / audio_duration_seconds, 4)
        est_90m_seconds = round(rtf * (90 * 60), 2)
        meets_30m_est = est_90m_seconds <= (30 * 60)

    return EvaluationReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        target_wer_percent=target_wer_percent,
        wer_percent=word_stats.error_rate_percent,
        cer_percent=char_stats.error_rate_percent,
        memenuhi_target_wer=word_stats.error_rate_percent <= target_wer_percent,
        word_alignment=word_stats,
        char_alignment=char_stats,
        audio_duration_seconds=audio_duration_seconds,
        processing_seconds=processing_seconds,
        real_time_factor=rtf,
        estimasi_kuliah_90_menit_detik=est_90m_seconds,
        memenuhi_target_30_menit_estimasi=meets_30m_est,
        kelengkapan_3_segmen_ringkasan=has_three_segments,
    )


def evaluate_audio_with_models(
    audio_path: Path,
    reference_text: str,
    *,
    model_sizes: Sequence[str],
    target_wer_percent: float = 20.0,
) -> list[EvaluationReport]:
    """Jalankan benchmark WER/CER/RTF pada satu file audio untuk beberapa ukuran model Whisper."""
    import os

    reports: list[EvaluationReport] = []
    original_model_size = os.environ.get("WHISPER_MODEL_SIZE")

    try:
        for size in model_sizes:
            os.environ["WHISPER_MODEL_SIZE"] = size
            LOGGER.info("Menguji model faster-whisper/%s pada %s...", size, audio_path)
            stt_res = transcribe_audio_detailed(audio_path, show_progress=False)
            report = evaluate_transcript(
                reference_text,
                stt_res.text,
                model_name=f"faster-whisper/{size}",
                target_wer_percent=target_wer_percent,
                audio_duration_seconds=stt_res.audio_duration_seconds,
                processing_seconds=stt_res.processing_seconds,
            )
            reports.append(report)
    finally:
        if original_model_size is None:
            os.environ.pop("WHISPER_MODEL_SIZE", None)
        else:
            os.environ["WHISPER_MODEL_SIZE"] = original_model_size

    return reports


def _print_report_table(reports: Sequence[EvaluationReport]) -> None:
    print("\n=== HASIL EVALUASI AKURASI SPEECH-TO-TEXT (NFR-07) ===")
    for rep in reports:
        status_wer = "LULUS" if rep.memenuhi_target_wer else "BELUM LULUS"
        print(f"\nModel                  : {rep.model_name}")
        print(f"WER (Word Error Rate)  : {rep.wer_percent:.2f}% (Target <= {rep.target_wer_percent:.2f}% -> {status_wer})")
        print(f"CER (Char Error Rate)  : {rep.cer_percent:.2f}%")
        wa = rep.word_alignment
        print(
            f"Rincian Kata           : Ref={wa.reference_length}, Hyp={wa.hypothesis_length}, "
            f"Sub={wa.substitutions}, Del={wa.deletions}, Ins={wa.insertions}"
        )
        if rep.real_time_factor is not None and rep.estimasi_kuliah_90_menit_detik is not None:
            status_30m = "LULUS" if rep.memenuhi_target_30_menit_estimasi else "MELEBIHI 30 MENIT"
            est_min = rep.estimasi_kuliah_90_menit_detik / 60.0
            print(
                f"Kinerja (RTF)          : {rep.real_time_factor:.4f} "
                f"(Proses {rep.processing_seconds:.2f}s untuk audio {rep.audio_duration_seconds:.2f}s)"
            )
            print(f"Estimasi Kuliah 90 Mnt : {est_min:.2f} menit ({status_30m})")
        if rep.kelengkapan_3_segmen_ringkasan is not None:
            seg_str = "LENGKAP (3 segmen)" if rep.kelengkapan_3_segmen_ringkasan else "TIDAK LENGKAP"
            print(f"Kelengkapan Ringkasan  : {seg_str}")
    print("=" * 55)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi akurasi Speech-to-Text (WER, CER, RTF) sesuai NFR-07 "
            "Proposal Capstone Box Archive."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path file teks transkrip referensi (ground truth) (.txt).",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--hypothesis",
        type=Path,
        help="Path file hasil transkripsi (.txt atau .json output dari main_pipeline.py).",
    )
    source_group.add_argument(
        "--audio",
        type=Path,
        help="Path file audio uji untuk langsung ditranskripsikan & dibandingkan dengan referensi.",
    )
    parser.add_argument(
        "--compare-models",
        nargs="+",
        default=["base"],
        help="Daftar ukuran model Whisper yang diuji jika memakai --audio (contoh: tiny base small).",
    )
    parser.add_argument(
        "--target-wer",
        type=float,
        default=20.0,
        help="Ambang batas target WER dalam persen untuk NFR-07 (default: 20.0).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Simpan laporan evaluasi dalam format JSON (opsional).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ref_path = args.reference.expanduser().resolve()
    if not ref_path.is_file():
        LOGGER.error("File referensi tidak ditemukan: %s", ref_path)
        return 1
    reference_text = ref_path.read_text(encoding="utf-8").strip()

    if args.hypothesis is not None:
        hyp_path = args.hypothesis.expanduser().resolve()
        if not hyp_path.is_file():
            LOGGER.error("File hipotesis tidak ditemukan: %s", hyp_path)
            return 1
        hyp_text, has_three_segments, model_name = _load_hypothesis_file(hyp_path)
        reports = [
            evaluate_transcript(
                reference_text,
                hyp_text,
                model_name=model_name,
                target_wer_percent=args.target_wer,
                has_three_segments=has_three_segments,
            )
        ]
    else:
        audio_path = args.audio.expanduser().resolve()
        if not audio_path.is_file():
            LOGGER.error("File audio tidak ditemukan: %s", audio_path)
            return 1
        reports = evaluate_audio_with_models(
            audio_path,
            reference_text,
            model_sizes=args.compare_models,
            target_wer_percent=args.target_wer,
        )

    _print_report_table(reports)

    if args.output_json is not None:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [rep.model_dump(mode="json") for rep in reports]
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Laporan evaluasi JSON tersimpan di: %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
