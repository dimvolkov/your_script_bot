import logging

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, WHISPER_MODEL, CHUNK_OVERLAP_SEC
from models.transcript import Segment, TranscriptResult
from services.youtube import get_chunk_offset

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def transcribe_file(file_path: str) -> tuple[list[Segment], str]:
    """Transcribe a single audio file via Whisper API."""
    with open(file_path, "rb") as f:
        response = await client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    for seg in response.segments:
        segments.append(Segment(start=seg["start"], end=seg["end"], text=seg["text"].strip()))

    return segments, response.language


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

        # Adjust timestamps with offset
        for seg in chunk_segments:
            seg.start += offset
            seg.end += offset

        if i > 0 and all_segments:
            # Remove overlap: drop segments from this chunk that fall within
            # the overlap zone (before midpoint of overlap)
            overlap_start = offset
            overlap_mid = offset + CHUNK_OVERLAP_SEC / 2
            chunk_segments = [s for s in chunk_segments if s.start >= overlap_mid]

            # Also trim any segments from previous chunk that go past the midpoint
            all_segments = [s for s in all_segments if s.end <= overlap_mid + 1]

        all_segments.extend(chunk_segments)

    full_text = " ".join(s.text for s in all_segments)
    return TranscriptResult(segments=all_segments, full_text=full_text, language=language)
