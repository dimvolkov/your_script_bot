import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

MAX_VIDEO_DURATION_HOURS = 3
CHUNK_SIZE_MB = 24
CHUNK_OVERLAP_SEC = 30
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/transcribator")
