import logging
import os
import re
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

from config import CHUNK_OVERLAP_SEC, CHUNK_SIZE_MB, MAX_VIDEO_DURATION_HOURS, TEMP_DIR

logger = logging.getLogger(__name__)

YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w\-]+"
)


def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.match(url.strip()))


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w\-]+)", url)
    return match.group(1) if match else ""


def download_audio(url: str, session_dir: str) -> tuple[str, str]:
    """Download audio from YouTube video. Returns (audio_path, video_title)."""
    os.makedirs(session_dir, exist_ok=True)
    output_path = os.path.join(session_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
        "postprocessor_args": ["-ac", "1"],
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "socket_timeout": 60,
        "retries": 3,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)
        if duration > MAX_VIDEO_DURATION_HOURS * 3600:
            raise ValueError(
                f"Video is too long ({duration // 3600}h). Max is {MAX_VIDEO_DURATION_HOURS}h."
            )
        title = info.get("title", "Untitled")
        ydl.download([url])

    # Find the downloaded file
    raw_path = os.path.join(session_dir, "audio.mp3")
    if not os.path.exists(raw_path):
        for f in os.listdir(session_dir):
            if f.startswith("audio."):
                raw_path = os.path.join(session_dir, f)
                break

    # Force re-encode to small file: 16kHz mono 32kbps
    import subprocess
    compressed_path = os.path.join(session_dir, "audio_small.mp3")
    subprocess.run(
        ["ffmpeg", "-i", raw_path, "-ac", "1", "-ar", "16000", "-ab", "24k",
         "-f", "mp3", compressed_path, "-y"],
        check=True, capture_output=True,
    )
    os.replace(compressed_path, os.path.join(session_dir, "audio.mp3"))
    # Remove original if different
    if raw_path != os.path.join(session_dir, "audio.mp3"):
        try:
            os.remove(raw_path)
        except OSError:
            pass

    audio_path = os.path.join(session_dir, "audio.mp3")
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f"Compressed audio: {size_mb:.1f} MB")

    return audio_path, title


MAX_CHUNK_DURATION_SEC = 1300  # Whisper API limit is 1400s, keep margin


def split_audio_if_needed(audio_path: str, session_dir: str) -> list[str]:
    """Split audio into chunks if it exceeds the size or duration limit. Returns list of chunk paths."""
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000.0

    needs_split_size = file_size_mb > CHUNK_SIZE_MB
    needs_split_duration = total_duration_sec > MAX_CHUNK_DURATION_SEC

    if not needs_split_size and not needs_split_duration:
        return [audio_path]

    logger.info(f"Audio is {file_size_mb:.1f}MB, {total_duration_sec:.0f}s, splitting into chunks...")

    # Determine chunk duration: respect both size and duration limits
    if needs_split_size:
        chunk_duration_by_size_ms = int(total_duration_ms * (CHUNK_SIZE_MB / file_size_mb))
    else:
        chunk_duration_by_size_ms = total_duration_ms

    chunk_duration_by_time_ms = MAX_CHUNK_DURATION_SEC * 1000
    chunk_duration_ms = min(chunk_duration_by_size_ms, chunk_duration_by_time_ms)
    overlap_ms = CHUNK_OVERLAP_SEC * 1000

    chunks = []
    start = 0
    chunk_index = 0

    while start < total_duration_ms:
        end = min(start + chunk_duration_ms, total_duration_ms)
        chunk = audio[start:end]

        chunk_path = os.path.join(session_dir, f"chunk_{chunk_index:03d}.mp3")
        chunk.export(chunk_path, format="mp3", parameters=["-ac", "1", "-ab", "64k"])
        chunks.append(chunk_path)

        logger.info(
            f"Chunk {chunk_index}: {start / 1000:.1f}s - {end / 1000:.1f}s "
            f"({os.path.getsize(chunk_path) / (1024 * 1024):.1f}MB)"
        )

        chunk_index += 1
        if end >= total_duration_ms:
            break
        start = end - overlap_ms

    return chunks


def get_chunk_offset(chunk_index: int, chunks: list[str], audio_path: str) -> float:
    """Calculate the time offset for a chunk, accounting for overlap."""
    if chunk_index == 0:
        return 0.0

    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if file_size_mb > CHUNK_SIZE_MB:
        chunk_duration_by_size_ms = int(total_duration_ms * (CHUNK_SIZE_MB / file_size_mb))
    else:
        chunk_duration_by_size_ms = total_duration_ms

    chunk_duration_by_time_ms = MAX_CHUNK_DURATION_SEC * 1000
    chunk_duration_ms = min(chunk_duration_by_size_ms, chunk_duration_by_time_ms)
    overlap_ms = CHUNK_OVERLAP_SEC * 1000

    offset_ms = chunk_index * (chunk_duration_ms - overlap_ms)
    return offset_ms / 1000.0


def get_session_dir(user_id: int) -> str:
    session_dir = os.path.join(TEMP_DIR, str(user_id))
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def extract_audio_from_file(video_path: str, session_dir: str) -> str:
    """Extract audio from a video file using ffmpeg. Returns audio path."""
    import subprocess
    audio_path = os.path.join(session_dir, "audio.mp3")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-ab", "24k", "-f", "mp3", audio_path, "-y"],
        check=True,
        capture_output=True,
    )
    return audio_path


def download_thumbnail(url: str, session_dir: str) -> str | None:
    """Download video thumbnail. Returns path or None on failure."""
    video_id = extract_video_id(url)
    if not video_id:
        return None

    import urllib.request
    thumb_path = os.path.join(session_dir, "thumbnail.jpg")
    for quality in ("maxresdefault", "hqdefault"):
        thumb_url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
        try:
            urllib.request.urlretrieve(thumb_url, thumb_path)
            if os.path.getsize(thumb_path) > 1000:
                return thumb_path
        except Exception:
            continue
    return None


def cleanup_session(session_dir: str) -> None:
    """Remove all files in the session directory."""
    if os.path.exists(session_dir):
        for f in os.listdir(session_dir):
            try:
                os.remove(os.path.join(session_dir, f))
            except OSError:
                pass
        try:
            os.rmdir(session_dir)
        except OSError:
            pass
