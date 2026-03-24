import json
import logging

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from models.transcript import AnalysisResult, Segment, TopicSection, TranscriptResult

logger = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client

ANALYSIS_PROMPT = """\
You are given a transcript of a YouTube video with timestamps.
Your task is to analyze it and produce a structured JSON response.

Instructions:
1. Write a concise summary (3-5 sentences) of the video content IN RUSSIAN.
2. Divide the transcript into logical topic sections (5-15 sections depending on length).
3. For each section provide: title (in Russian), start_time, end_time, content (the transcript text for that section, translated to Russian), and action_steps — a list of concrete, practical step-by-step instructions that a viewer should follow based on this section's content. Each step should be a clear actionable instruction in Russian.
4. If the original language is already Russian, keep the text as-is but still structure it into sections.

IMPORTANT:
- Respond ONLY with valid JSON in this exact format.
- Do NOT use literal newlines inside string values. Use spaces instead.
- Make sure all JSON strings are properly escaped.
{
  "summary": "...",
  "sections": [
    {
      "title": "Section title in Russian",
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "content": "Translated/original text of this section in Russian",
      "action_steps": ["Шаг 1: ...", "Шаг 2: ...", "..."]
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

    response = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16384,
        messages=[{"role": "user", "content": full_prompt}],
    )

    raw_text = response.content[0].text
    try:
        return _parse_response(raw_text)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse Claude response, asking Claude to fix JSON...")
        data = await _fix_json_with_claude(raw_text)
        return _parse_response_from_dict(data)


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

    response = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": toc_prompt}],
    )

    toc_raw = response.content[0].text
    try:
        toc_data = _parse_json(toc_raw)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse TOC JSON, asking Claude to fix...")
        toc_data = await _fix_json_with_claude(toc_raw)
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


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _parse_json(text: str) -> dict:
    text = _strip_markdown_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object if surrounded by extra text
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from Claude response")


async def _fix_json_with_claude(broken_json: str) -> dict:
    """Ask Claude to fix broken JSON."""
    response = await _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16384,
        messages=[{"role": "user", "content": (
            "The following JSON is malformed. Fix it and return ONLY valid JSON, nothing else. "
            "Do not use literal newlines inside string values. "
            "Do not wrap in markdown code blocks.\n\n"
            + broken_json
        )}],
    )
    return _parse_json(response.content[0].text)


def _parse_response_from_dict(data: dict) -> AnalysisResult:
    sections = []
    for sec in data.get("sections", []):
        sections.append(
            TopicSection(
                title=sec.get("title", ""),
                start_time=sec.get("start_time", ""),
                end_time=sec.get("end_time", ""),
                content=sec.get("content", ""),
                action_steps=sec.get("action_steps", []),
            )
        )
    return AnalysisResult(summary=data.get("summary", ""), sections=sections)


def _parse_response(text: str) -> AnalysisResult:
    data = _parse_json(text)
    return _parse_response_from_dict(data)
