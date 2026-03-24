import asyncio
import logging
import os

from aiogram import Router, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile

from services.youtube import (
    is_valid_youtube_url,
    download_audio,
    download_thumbnail,
    extract_audio_from_file,
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
    "Я транскрибирую видео и создаю структурированный документ на русском языке.\n\n"
    "Отправьте мне:\n"
    "• Ссылку на YouTube видео\n"
    "• Видеофайл из Телеграм\n"
    "• Видеокружок\n\n"
    "Я:\n"
    "1. Извлеку аудио\n"
    "2. Транскрибирую через Whisper\n"
    "3. Переведу и структурирую через Claude\n"
    "4. Отправлю .docx файл с саммари, оглавлением и полным транскриптом\n\n"
    "Ограничения: видео до 3 часов.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/transcribe <url> — транскрибировать видео\n"
    "/balance — проверить статус API ключей"
)


@router.message(Command("balance"))
async def cmd_balance(message: Message) -> None:
    import httpx
    import socket
    from config import OPENAI_API_KEY, ANTHROPIC_API_KEY

    lines = []

    # Network diagnostics
    for host in ("api.openai.com", "api.anthropic.com"):
        try:
            ip = socket.getaddrinfo(host, 443)[0][4][0]
            lines.append(f"🌐 {host} → {ip}")
        except Exception as e:
            lines.append(f"🌐 {host} → DNS ошибка: {e}")

    # OpenAI key check
    lines.append(f"🔑 OpenAI ключ: {'задан' if OPENAI_API_KEY else 'НЕ задан'} ({OPENAI_API_KEY[:8]}...)" if OPENAI_API_KEY else "🔑 OpenAI ключ: НЕ задан")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            resp = await client.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                params={"limit": 1},
            )
            if resp.status_code == 200:
                lines.append("🤖 OpenAI: ключ активен ✅")
            elif resp.status_code == 401:
                lines.append("🤖 OpenAI: ключ недействителен ❌")
            elif resp.status_code == 429:
                lines.append("🤖 OpenAI: превышен лимит запросов ⚠️")
            else:
                lines.append(f"🤖 OpenAI: статус {resp.status_code}")
    except Exception as e:
        lines.append(f"🤖 OpenAI: ошибка — {type(e).__name__}: {e}")

    # Anthropic key check
    lines.append(f"🔑 Anthropic ключ: {'задан' if ANTHROPIC_API_KEY else 'НЕ задан'}")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
            if resp.status_code == 200:
                lines.append("🧠 Anthropic: ключ активен ✅")
            elif resp.status_code == 401:
                lines.append("🧠 Anthropic: ключ недействителен ❌")
            elif resp.status_code == 429:
                lines.append("🧠 Anthropic: превышен лимит запросов ⚠️")
            else:
                lines.append(f"🧠 Anthropic: статус {resp.status_code}")
    except Exception as e:
        lines.append(f"🧠 Anthropic: ошибка — {type(e).__name__}: {e}")

    await message.answer("\n".join(lines))


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
    await _process_youtube(message, url)


@router.message(F.video | F.video_note | F.document)
async def handle_telegram_video(message: Message) -> None:
    if message.document:
        mime = message.document.mime_type or ""
        if not mime.startswith("video/"):
            await message.answer(
                "Отправьте ссылку на YouTube видео, видеофайл или видеокружок."
            )
            return

    await _process_telegram_video(message)


@router.message(F.text)
async def handle_url(message: Message) -> None:
    url = message.text.strip()
    if is_valid_youtube_url(url):
        await _process_youtube(message, url)
    elif "t.me/" in url or "telegram." in url:
        await message.answer(
            "Ссылки на Telegram видео не поддерживаются.\n"
            "Перешлите видео напрямую в этот чат — просто сделайте Forward сообщения с видео."
        )
    else:
        await message.answer(
            "Отправьте ссылку на YouTube видео или перешлите видеофайл/видеокружок в этот чат."
        )


async def _check_active(message: Message) -> tuple[int, str] | None:
    """Check if user already has an active request. Returns (user_id, session_dir) or None."""
    user_id = message.from_user.id
    if user_id in _active_users:
        await message.answer("Подождите завершения предыдущего запроса.")
        return None
    _active_users.add(user_id)
    return user_id, get_session_dir(user_id)


async def _process_youtube(message: Message, url: str) -> None:
    if not is_valid_youtube_url(url):
        await message.answer("Некорректная ссылка на YouTube видео.")
        return

    result = await _check_active(message)
    if not result:
        return
    user_id, session_dir = result
    status_msg = await message.answer("⏳ [1/4] Скачиваю аудио...")

    try:
        audio_path, title = await asyncio.to_thread(download_audio, url, session_dir)
        chunks = await asyncio.to_thread(split_audio_if_needed, audio_path, session_dir)

        transcript, analysis = await _transcribe_and_analyze(status_msg, chunks, audio_path)

        await status_msg.edit_text("📄 [4/4] Создаю документ...")
        thumbnail_path = await asyncio.to_thread(download_thumbnail, url, session_dir)
        docx_path = await asyncio.to_thread(
            generate_docx, title, analysis, session_dir,
            video_url=url, thumbnail_path=thumbnail_path,
        )

        doc_file = FSInputFile(docx_path, filename=os.path.basename(docx_path))
        await message.answer_document(doc_file, caption=f"Транскрипт: {title}")
        await status_msg.delete()

    except ValueError as e:
        await status_msg.edit_text(f"Ошибка: {e}")
    except Exception as e:
        logger.exception("Error processing YouTube video")
        cause = e.__cause__ or e.__context__
        err_info = f"{type(e).__name__}: {e}"
        if cause:
            err_info += f"\n\nCaused by: {type(cause).__name__}: {cause}"
        await status_msg.edit_text(f"Ошибка:\n\n<code>{err_info[:900]}</code>", parse_mode="HTML")
    finally:
        _active_users.discard(user_id)
        cleanup_session(session_dir)


async def _process_telegram_video(message: Message) -> None:
    result = await _check_active(message)
    if not result:
        return
    user_id, session_dir = result
    status_msg = await message.answer("⏳ [1/4] Скачиваю видео...")

    try:
        # Get file from Telegram
        if message.video:
            file_obj = message.video
            title = message.caption or "Telegram видео"
        elif message.video_note:
            file_obj = message.video_note
            title = "Видеокружок"
        else:
            file_obj = message.document
            title = message.caption or message.document.file_name or "Видео"

        file = await message.bot.get_file(file_obj.file_id)
        video_path = os.path.join(session_dir, "video" + os.path.splitext(file.file_path or ".mp4")[1])
        await message.bot.download_file(file.file_path, video_path)

        # Extract audio
        await status_msg.edit_text("⏳ [1/4] Извлекаю аудио...")
        audio_path = await asyncio.to_thread(extract_audio_from_file, video_path, session_dir)
        chunks = await asyncio.to_thread(split_audio_if_needed, audio_path, session_dir)

        transcript, analysis = await _transcribe_and_analyze(status_msg, chunks, audio_path)

        await status_msg.edit_text("📄 [4/4] Создаю документ...")
        docx_path = await asyncio.to_thread(
            generate_docx, title, analysis, session_dir,
        )

        doc_file = FSInputFile(docx_path, filename=os.path.basename(docx_path))
        await message.answer_document(doc_file, caption=f"Транскрипт: {title}")
        await status_msg.delete()

    except TelegramBadRequest as e:
        if "file is too big" in str(e):
            await status_msg.edit_text(
                "Файл слишком большой (лимит Telegram — 20 МБ).\n"
                "Попробуйте отправить видео меньшего размера или загрузить его на YouTube и прислать ссылку."
            )
        else:
            await status_msg.edit_text(f"Ошибка Telegram: {e}")
    except ValueError as e:
        await status_msg.edit_text(f"Ошибка: {e}")
    except Exception as e:
        logger.exception("Error processing Telegram video")
        import traceback
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        short_tb = "".join(tb[-3:])[:800]
        await status_msg.edit_text(f"Ошибка:\n\n<code>{short_tb}</code>", parse_mode="HTML")
    finally:
        _active_users.discard(user_id)
        cleanup_session(session_dir)


async def _transcribe_and_analyze(status_msg, chunks, audio_path):
    """Shared steps 2-3: transcribe and analyze."""
    await status_msg.edit_text("🎙 [2/4] Транскрибирую аудио...")
    transcript = await transcribe_audio(chunks, audio_path)

    await status_msg.edit_text("🧠 [3/4] Анализирую и перевожу...")
    analysis = await analyze_transcript(transcript)

    return transcript, analysis
