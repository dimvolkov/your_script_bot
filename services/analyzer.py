import json
import logging

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from models.transcript import AnalysisResult, Segment, TopicSection, TranscriptResult

logger = logging.getLogger(__name__)

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

ANALYSIS_PROMPT = """\
You are given a transcript of a YouTube video with timestamps.
Your task is to analyze it and produce a structured JSON response.

Instructions:
1. Write a concise summary (3-5 sentences) of the video content IN RUSSIAN.
2. Divide the transcript into logical topic sections (5-15 sections depending on length).
3. For each section provide: title (in Russian), start_time, end_time, and content (the transcript text for that section, translated to Russian).
4. If the original language is already Russian, keep the text as-is but still structure it into sections.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "summary": "...",
  "sections": [
    {
      "title": "Section title in Russian",
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "content": "Translated/original text of this section in Russian"
    }
  ]
}

Here is the transcript:
"""


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def format_transcript_with_timestamps(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        ts = format_timestamp(seg.start)
        lines.append(f"[{ts}] {seg.text}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


async def analyze_transcript(transcript: TranscriptResult) -> AnalysisResult:
    """Send transcript to Claude for analysis, translation, and structuring."""
    formatted = format_transcript_with_timestamps(transcript.segments)
    full_prompt = ANALYSIS_PROMPT + formatted

    token_estimate = estimate_tokens(full_prompt)
    logger.info(f"Estimated prompt tokens: {token_estimate}")

    if token_estimate > 120_000:
        return await _analyze_long_transcript(transcript)

    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": full_prompt}],
    )

    return _parse_response(response.content[0].text)


async def _analyze_long_transcript(transcript: TranscriptResult) -> AnalysisResult:
    """Handle very long transcripts by splitting into two requests."""
    formatted = format_transcript_with_timestamps(transcript.segments)

    # First request: summary + section boundaries
    toc_prompt = f"""\
You are given a very long transcript of a YouTube video with timestamps.
Produce a JSON with:
1. "summary": concise summary in Russian (3-5 sentences)
2. "sections": list of section boundaries with "title" (Russian), "start_time" (MM:SS), "end_time" (MM:SS)
   Do NOT include "content" yet — just titles and time boundaries.

Respond ONLY with valid JSON.

Transcript (first 30000 chars):
{formatted[:30000]}

... (transcript continues, total length: {len(formatted)} chars)

Full timestamps range: {format_timestamp(transcript.segments[0].start)} - {format_timestamp(transcript.segments[-1].end)}
"""

    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": toc_prompt}],
    )

    toc_data = _parse_json(response.content[0].text)
    summary = toc_data.get("summary", "")
    section_boundaries = toc_data.get("sections", [])

    # Second request: fill in content for each section
    sections = []
    for sec in section_boundaries:
        start_time = sec.get("start_time", "00:00")
        end_time = sec.get("end_time", "00:00")

        # Extract relevant segment text
        start_sec = _parse_time(start_time)
        end_sec = _parse_time(end_time)
        section_text = " ".join(
            s.text for s in transcript.segments if start_sec <= s.start < end_sec
        )

        if not section_text:
            continue

        content_prompt = f"""\
Translate the following transcript excerpt to Russian. Keep it natural and readable.
If already in Russian, clean it up for readability.
Return ONLY the translated text, no JSON, no extra formatting.

Text:
{section_text[:15000]}
"""
        content_response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": content_prompt}],
        )

        sections.append(
            TopicSection(
                title=sec.get("title", ""),
                start_time=start_time,
                end_time=end_time,
                content=content_response.content[0].text.strip(),
            )
        )

    return AnalysisResult(summary=summary, sections=sections)


def _parse_time(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def _parse_response(text: str) -> AnalysisResult:
    data = _parse_json(text)

    sections = []
    for sec in data.get("sections", []):
        sections.append(
            TopicSection(
                title=sec.get("title", ""),
                start_time=sec.get("start_time", ""),
                end_time=sec.get("end_time", ""),
                content=sec.get("content", ""),
            )
        )

    return AnalysisResult(summary=data.get("summary", ""), sections=sections)
