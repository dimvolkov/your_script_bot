import asyncio
import logging
import os
import time

import requests

from config import OPENAI_API_KEY, WHISPER_MODEL, CHUNK_OVERLAP_SEC
from models.transcript import Segment, TranscriptResult
from services.youtube import get_chunk_offset

logger = logging.getLogger(__name__)

WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"
MAX_RETRIES = 3


def _call_whisper_sync(file_path: str) -> dict:
    """Synchronous Whisper API call using requests — runs in a thread."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    WHISPER_API_URL,
                    headers=headers,
                    files={"file": (os.path.basename(file_path), f, "audio/mpeg")},
                    data={
                        "model": WHISPER_MODEL,
                        "response_format": "verbose_json",
                    },
                    timeout=600,
                )

            if resp.status_code == 500:
                logger.warning(f"Whisper attempt {attempt}/{MAX_RETRIES}: 500 — {resp.text[:200]}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)
                    continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Whisper API error {resp.status_code}: {resp.text[:800]}"
                )
            return resp.json()

        except requests.RequestException as e:
            last_error = e
            logger.warning(f"Whisper attempt {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Whisper API failed after {MAX_RETRIES} attempts: {last_error}")


async def transcribe_file(file_path: str) -> tuple[list[Segment], str]:
    """Transcribe a single audio file via Whisper API."""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Transcribing {file_path} ({size_mb:.1f} MB)...")

    result = await asyncio.to_thread(_call_whisper_sync, file_path)

    segments = []
    for seg in result.get("segments", []):
        segments.append(Segment(start=seg["start"], end=seg["end"], text=seg["text"].strip()))

    return segments, result.get("language", "")


async def transcribe_audio(
    chunks: list[str], original_audio_path: str
) -> TranscriptResult:
    """Transcribe audio chunks and merge results."""
    if len(chunks) == 1:
        segments, language = await transcribe_file(chunks[0])
        full_text = " ".join(s.text for s in segments)
        return TranscriptResult(segments=segments, full_text=full_text, language=language)

    all_segments: list[Segment] = []
    language = ""

    for i, chunk_path in enumerate(chunks):
        logger.info(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        chunk_segments, chunk_lang = await transcribe_file(chunk_path)

        if i == 0:
            language = chunk_lang

        offset = get_chunk_offset(i, chunks, original_audio_path)

        for seg in chunk_segments:
            seg.start += offset
            seg.end += offset

        if i > 0 and all_segments:
            overlap_mid = offset + CHUNK_OVERLAP_SEC / 2
            chunk_segments = [s for s in chunk_segments if s.start >= overlap_mid]
            all_segments = [s for s in all_segments if s.end <= overlap_mid + 1]

        all_segments.extend(chunk_segments)

    full_text = " ".join(s.text for s in all_segments)
    return TranscriptResult(segments=all_segments, full_text=full_text, language=language)
