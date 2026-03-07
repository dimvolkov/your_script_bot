import asyncio
import logging
import os

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile

from services.youtube import (
    is_valid_youtube_url,
    download_audio,
    download_thumbnail,
    split_audio_if_needed,
    get_session_dir,
    cleanup_session,
)
from services.transcriber import transcribe_audio
from services.analyzer import analyze_transcript
from services.document import generate_docx

logger = logging.getLogger(__name__)
router = Router()

# Track active users to allow only one request at a time
_active_users: set[int] = set()

HELP_TEXT = (
    "Я транскрибирую YouTube видео и создаю структурированный документ на русском языке.\n\n"
    "Просто отправьте мне ссылку на YouTube видео, и я:\n"
    "1. Скачаю аудио\n"
    "2. Транскрибирую через Whisper\n"
    "3. Переведу и структурирую через Claude\n"
    "4. Отправлю .docx файл с саммари, оглавлением и полным транскриптом\n\n"
    "Ограничения: видео до 3 часов.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/transcribe <url> — транскрибировать видео"
)


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Отправьте мне ссылку на YouTube видео, "
        "и я создам структурированный транскрипт на русском языке в формате .docx."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("transcribe"))
async def cmd_transcribe(message: Message) -> None:
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Использование: /transcribe <youtube_url>")
        return
    url = parts[1].strip()
    await _process_video(message, url)


@router.message(F.text)
async def handle_url(message: Message) -> None:
    url = message.text.strip()
    if is_valid_youtube_url(url):
        await _process_video(message, url)
    else:
        await message.answer(
            "Отправьте ссылку на YouTube видео или используйте /help для справки."
        )


async def _process_video(message: Message, url: str) -> None:
    user_id = message.from_user.id

    if not is_valid_youtube_url(url):
        await message.answer("Некорректная ссылка на YouTube видео.")
        return

    if user_id in _active_users:
        await message.answer("Подождите завершения предыдущего запроса.")
        return

    _active_users.add(user_id)
    session_dir = get_session_dir(user_id)
    status_msg = await message.answer("⏳ [1/4] Скачиваю аудио...")

    try:
        # Step 1: Download audio
        audio_path, title = await asyncio.to_thread(
            download_audio, url, session_dir
        )

        # Step 1.5: Split if needed
        chunks = await asyncio.to_thread(split_audio_if_needed, audio_path, session_dir)

        # Step 2: Transcribe
        await status_msg.edit_text("🎙 [2/4] Транскрибирую аудио...")
        transcript = await transcribe_audio(chunks, audio_path)

        # Step 3: Analyze
        await status_msg.edit_text("🧠 [3/4] Анализирую и перевожу...")
        analysis = await analyze_transcript(transcript)

        # Step 4: Generate document
        await status_msg.edit_text("📄 [4/4] Создаю документ...")
        thumbnail_path = await asyncio.to_thread(download_thumbnail, url, session_dir)
        docx_path = await asyncio.to_thread(
            generate_docx, title, analysis, session_dir,
            video_url=url, thumbnail_path=thumbnail_path,
        )

        # Send document
        doc_file = FSInputFile(docx_path, filename=os.path.basename(docx_path))
        await message.answer_document(doc_file, caption=f"Транскрипт: {title}")
        await status_msg.delete()

    except ValueError as e:
        await status_msg.edit_text(f"Ошибка: {e}")
    except Exception:
        logger.exception("Error processing video")
        await status_msg.edit_text(
            "Произошла ошибка при обработке видео. Попробуйте позже."
        )
    finally:
        _active_users.discard(user_id)
        cleanup_session(session_dir)
