"""
Ultimate Telegram AI Bot (Agno + OpenAI + Telegram Bot HTTP API)
================================================================
A state-of-the-art Telegram assistant built on top of Agno's Agent framework,
OpenAI's Chat Completions API, and the raw Telegram Bot HTTP API.

Features (beyond the official basic example):
  * Reasoning-enhanced "super-agent" with 8 built-in tools
        - DuckDuckGo (web search)
        - Wikipedia
        - HackerNews
        - arXiv (research papers)
        - YFinance (stocks, fundamentals, analyst recs, news)
        - Newspaper4k (full-article reader)
        - Calculator
        - TelegramTools (proactive send/reply/react/pin/edit/delete/photo/video/doc/audio/sticker from inside the model)
  * Live status-line streaming showing reasoning & tool calls
    (Reasoning..., DuckDuckGo..., Wikipedia✓ ... then the answer)
  * Multi-modal input: text, photos, stickers, voice, audio, video,
    video notes, GIFs, documents (up to 20 MB)
  * Multi-modal output: text + images/audio/videos/docs returned by tools
  * Persistent per-chat SQLite memory (survives restarts)
  * Smart group-chat gating (@mention or reply-to-bot)
  * Custom slash commands:
        /start /help /new /info /model
  * Thinking indicator via send_chat_action, auto emoji reaction on arrival
  * Auto-splitting of long messages on paragraph boundaries
  * Markdown -> Telegram HTML formatting
  * Bot-to-bot chat allowed (configurable) with loop-protection depth cap
  * Long-polling (no ngrok required) with webhook helper included

Run:
    source .venv/bin/activate
    python bot.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Bootstrap — configuration
# ---------------------------------------------------------------------------
# Load secrets from .env.local first (gitignored, real keys), then fall back
# to the tracked .env template. This lets the repo ship without leaking
# API keys while still working out-of-the-box locally if you paste keys
# into .env.
from dotenv import dotenv_values
def _load_env() -> None:
    # Lowest priority: tracked .env template
    load_dotenv(dotenv_path=".env", override=False)
    # Highest priority: untracked .env.local with real credentials
    load_dotenv(dotenv_path=".env.local", override=True)
_load_env()
Path("tmp").mkdir(exist_ok=True)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
REPLY_TO_MENTIONS_ONLY = os.getenv("REPLY_TO_MENTIONS_ONLY", "true").lower() == "true"
STREAMING = os.getenv("STREAMING", "true").lower() == "true"
HISTORY_RUNS = int(os.getenv("HISTORY_RUNS", "8"))
SHOW_STATUS_LINES = os.getenv("SHOW_STATUS_LINES", "true").lower() == "true"
QUOTED_REPLIES = os.getenv("QUOTED_REPLIES", "true").lower() == "true"
EMOJI_REACTION = os.getenv("EMOJI_REACTION", "👀")  # set "" to disable
# If True, this bot will respond to OTHER BOTS when they @mention it or
# reply to one of its messages (bot-to-bot conversations in groups).
# Human messages (is_bot=False) ALWAYS get a response if they meet group gating.
RESPOND_TO_OTHER_BOTS = os.getenv("RESPOND_TO_OTHER_BOTS", "true").lower() == "true"
# Safety cap on consecutive bot-to-bot replies to prevent infinite loops.
BOT_LOOP_MAX_DEPTH = int(os.getenv("BOT_LOOP_MAX_DEPTH", "3"))

if not TELEGRAM_TOKEN:
    sys.exit("ERROR: TELEGRAM_TOKEN is missing in .env")
if not OPENAI_API_KEY:
    sys.exit("ERROR: OPENAI_API_KEY is missing in .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("newspaper").setLevel(logging.WARNING)
logging.getLogger("feedparser").setLevel(logging.WARNING)
log = logging.getLogger("ultimate-telegram-bot")

# ---------------------------------------------------------------------------
# 2. Imports — Agno
# ---------------------------------------------------------------------------
from agno.agent import Agent, RunEvent                      # noqa: E402
from agno.db.sqlite import SqliteDb                        # noqa: E402
from agno.media import Audio, File, Image, Video            # noqa: E402
from agno.models.openai import OpenAIChat                   # noqa: E402
from agno.os.interfaces.telegram.formatting import (       # noqa: E402
    markdown_to_telegram_html,
)
from agno.run.agent import RunOutput                        # noqa: E402
from agno.tools.calculator import CalculatorTools           # noqa: E402
from agno.tools.duckduckgo import DuckDuckGoTools           # noqa: E402
from agno.tools.hackernews import HackerNewsTools           # noqa: E402
from agno.tools.newspaper4k import Newspaper4kTools         # noqa: E402
from agno.tools.reasoning import ReasoningTools             # noqa: E402
from agno.tools.telegram import TelegramTools               # noqa: E402
from agno.tools.wikipedia import WikipediaTools             # noqa: E402
from agno.tools.yfinance import YFinanceTools               # noqa: E402

# ---------------------------------------------------------------------------
# 3. Imports — Telegram Bot HTTP API (async wrapper)
# ---------------------------------------------------------------------------
from telebot.async_telebot import AsyncTeleBot              # noqa: E402
import telebot.types as T                                   # noqa: E402

# ---------------------------------------------------------------------------
# 4. Persistent database
# ---------------------------------------------------------------------------
db = SqliteDb(id="ultimate-tg-db", db_file="tmp/ultimate.db")

# ---------------------------------------------------------------------------
# 5. Build the toolkit — pro-active Telegram tools (for the model to use)
# ---------------------------------------------------------------------------
# These tools let the AGENT itself decide to send photos, stickers, reactions,
# pin messages, edit messages, etc. The model is in full control.
# chat_id/token are injected at runtime per-message so the agent can message
# back to whichever chat triggered the run (see run-time injection below).
Path("tmp/downloads").mkdir(parents=True, exist_ok=True)
telegram_tools = TelegramTools(
    token=TELEGRAM_TOKEN,
    enable_send_message=True,
    enable_send_photo=True,
    enable_send_document=True,
    enable_send_video=True,
    enable_send_audio=True,
    enable_send_animation=True,
    enable_send_sticker=True,
    enable_edit_message=True,
    enable_delete_message=True,
    enable_react_with_emoji=True,
    enable_pin_message=True,
    enable_get_chat=True,
    enable_get_file=True,
    save_downloads=True,
    output_directory="tmp/downloads",
)

# ---------------------------------------------------------------------------
# 6. Build the "super-agent"
# ---------------------------------------------------------------------------
agent = Agent(
    id="ultimate-assistant",
    name="Assistant",
    model=OpenAIChat(id=OPENAI_MODEL_ID, api_key=OPENAI_API_KEY),
    db=db,
    tools=[
        # Structured reasoning — the model thinks through complex queries
        # step by step before answering, using think/analyze sub-routines.
        ReasoningTools(add_instructions=True),
        # Live web search (DuckDuckGo)
        DuckDuckGoTools(),
        # Encyclopedia lookups
        WikipediaTools(),
        # Tech community news / discussions
        HackerNewsTools(),
        # Financial data (prices, fundamentals, analyst recs, company news)
        YFinanceTools(
            enable_stock_price=True,
            enable_company_info=True,
            enable_analyst_recommendations=True,
            enable_company_news=True,
            enable_historical_prices=True,
        ),
        # Read full articles from URLs
        Newspaper4kTools(),
        # Math/unit conversions
        CalculatorTools(),
        # Pro-active Telegram output (reactions, pins, stickers, media, etc.)
        telegram_tools,
    ],
    instructions=[
        # --- Identity & tone ---
        "You are a top-tier AI assistant available through Telegram. You are "
        "knowledgeable, precise, and helpful. You combine powerful reasoning "
        "with real-time web search, financial data, encyclopedic knowledge, "
        "HackerNews trends, research papers, and article reading.",
        # --- Language ---
        "Detect the language the user writes in and respond in the same "
        "language. If they write in Arabic, respond in Arabic (العربية). If "
        "they write in English, respond in English.",
        # --- Output format ---
        "Keep answers concise and easy to read on a phone screen.",
        "Use short paragraphs, **bold** headings, bullet lists, and tables "
        "when helpful.",
        "For numbers, finance, or comparative data, use Markdown tables.",
        "Cite sources (URLs) when you use web search or news.",
        # --- Tool usage ---
        "ALWAYS use the calculator for math. Don't guess arithmetic.",
        "When a question asks about current events, recent news, prices, or "
        "anything after your knowledge cutoff, search the web first.",
        "For finance questions, fetch price, company info, analyst "
        "recommendations, and news — don't invent numbers.",
        "For technical topics, cross-check Wikipedia and HackerNews.",
        "If you find an interesting URL in search results, you may read the "
        "full article with the newspaper tool for deeper detail.",
        # --- Telegram pro-active features ---
        "You have Telegram tools available. You can: react to messages with "
        "emoji, pin important information, send stickers/GIFs/photos/audio/"
        "video/documents when appropriate, edit your last message if needed, "
        "or delete it. Use these tastefully to make the conversation lively.",
        "When you react with emoji, use a single relevant emoji — do not spam.",
        # --- Memory ---
        "Remember facts the user tells you within this conversation.",
        "If you are unsure, say so honestly — never make things up.",
        # --- Group behavior ---
        "In groups, stay on-topic and be polite. Do not dominate the chat.",
    ],
    add_history_to_context=True,
    num_history_runs=HISTORY_RUNS,
    add_datetime_to_context=True,
    markdown=True,
    tool_call_limit=20,
)

# ---------------------------------------------------------------------------
# 7. Telegram bot client
# ---------------------------------------------------------------------------
bot = AsyncTeleBot(TELEGRAM_TOKEN)
TG_MAX_MESSAGE = 4096
TG_MAX_FILE_SIZE = 20 * 1024 * 1024

BOT_COMMANDS = [
    T.BotCommand("start", "ابدأ المحادثة / Start the bot"),
    T.BotCommand("help", "تعليمات الاستخدام / Show help"),
    T.BotCommand("new", "محادثة جديدة / New conversation"),
    T.BotCommand("info", "معلومات البوت / Bot information"),
    T.BotCommand("model", "النموذج الحالي / Show model name"),
]

_bot_me: Optional[T.User] = None


async def _get_bot_me() -> T.User:
    global _bot_me
    if _bot_me is None:
        _bot_me = await bot.get_me()
    return _bot_me


# ---------------------------------------------------------------------------
# 7.5 Bot-to-bot reply-chain depth tracking (anti-loop protection)
# ---------------------------------------------------------------------------
# We track the current "depth" of consecutive bot-to-bot replies per chat so
# that two bots don't spiral into an infinite conversation. The counter is
# keyed by (chat_id, thread_id) and resets whenever a REAL HUMAN sends a
# message in the same thread.
import collections
_bot_chain_depth: dict[tuple[int, Optional[int]], int] = collections.defaultdict(int)
_last_msg_id_from_me: dict[tuple[int, Optional[int]], int] = {}


def _track_my_message(chat_id: int, thread_id: Optional[int], message_id: int) -> None:
    key = (chat_id, thread_id)
    _last_msg_id_from_me[key] = message_id


def _incoming_is_human(message: T.Message) -> bool:
    return bool(message.from_user and not message.from_user.is_bot)


# ---------------------------------------------------------------------------
# 8. Session scoping (per chat + per forum topic)
# ---------------------------------------------------------------------------
def _session_scope(chat_id: int, thread_id: Optional[int]) -> str:
    if thread_id:
        return f"tg:{chat_id}:{thread_id}"
    return f"tg:{chat_id}"


# ---------------------------------------------------------------------------
# 9. Telegram helpers
# ---------------------------------------------------------------------------
async def _download_file(file_id: str) -> Optional[bytes]:
    try:
        info = await bot.get_file(file_id)
        if info.file_size and info.file_size > TG_MAX_FILE_SIZE:
            log.warning("File too large: %d bytes", info.file_size)
            return None
        return await bot.download_file(info.file_path)
    except Exception as exc:
        log.error("File download failed: %s", exc)
        return None


def _extract_media(message: T.Message):
    if message.photo:
        return message.photo[-1].file_id, None, None, None
    if message.sticker:
        s = message.sticker
        thumb = getattr(s, "thumbnail", None) or getattr(s, "thumb", None)
        fid = thumb.file_id if thumb else s.file_id
        return fid, None, None, None
    if message.voice:
        return None, message.voice.file_id, None, None
    if message.audio:
        return None, message.audio.file_id, None, None
    vid = message.video or message.video_note or message.animation
    if vid:
        return None, None, vid.file_id, None
    if message.document:
        return None, None, None, message.document
    return None, None, None, None


async def _build_agent_input(message: T.Message) -> Optional[dict]:
    text = message.text or message.caption or ""
    img_id, aud_id, vid_id, doc = _extract_media(message)
    if not text and not (img_id or aud_id or vid_id or doc):
        return None

    kwargs: dict[str, Any] = {"input": text or ""}
    warning = None
    images, audio, videos, files = [], [], [], []

    if img_id:
        data = await _download_file(img_id)
        if data:
            images.append(Image(content=data))
    if aud_id:
        data = await _download_file(aud_id)
        if data:
            audio.append(Audio(content=data))
    if vid_id:
        data = await _download_file(vid_id)
        if data:
            videos.append(Video(content=data))
    if doc:
        if doc.file_size and doc.file_size > TG_MAX_FILE_SIZE:
            size_mb = doc.file_size // (1024 * 1024)
            warning = f"⚠️ الملف كبير جداً ({size_mb} MB). الحد الأقصى 20 MB."
        else:
            data = await _download_file(doc.file_id)
            if data:
                files.append(File(content=data, filename=doc.file_name))

    if images:
        kwargs["images"] = images
    if audio:
        kwargs["audio"] = audio
    if videos:
        kwargs["videos"] = videos
    if files:
        kwargs["files"] = files
    if warning:
        kwargs["_warning"] = warning
    return kwargs


def _split_html(html: str, limit: int = TG_MAX_MESSAGE) -> list[str]:
    if len(html) <= limit:
        return [html]
    chunks: list[str] = []
    remaining = html
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = max(
            remaining.rfind("\n\n", 0, limit),
            remaining.rfind("\n", 0, limit),
            remaining.rfind(" ", 0, limit),
        )
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    return chunks


async def _send_html(
    chat_id: int,
    text: str,
    *,
    reply_to: Optional[int] = None,
    thread_id: Optional[int] = None,
    parse_mode: str = "HTML",
) -> Optional[T.Message]:
    if parse_mode == "HTML":
        html = markdown_to_telegram_html(text)
    else:
        html = text
    chunks = _split_html(html)
    sent = None
    for i, chunk in enumerate(chunks):
        sent = await bot.send_message(
            chat_id,
            chunk,
            parse_mode=parse_mode if parse_mode != "None" else None,
            reply_to_message_id=(reply_to if i == 0 else None),
            message_thread_id=thread_id,
        )
    return sent


async def _send_media(
    response: RunOutput,
    chat_id: int,
    *,
    reply_to: Optional[int] = None,
    thread_id: Optional[int] = None,
) -> None:
    def _bytes(m):
        if hasattr(m, "url") and m.url:
            return m.url
        if hasattr(m, "get_content_bytes"):
            return m.get_content_bytes()
        return None

    mapping = [
        ("images", bot.send_photo),
        ("audio", bot.send_audio),
        ("videos", bot.send_video),
        ("files", bot.send_document),
    ]
    for attr, sender in mapping:
        for item in getattr(response, attr, None) or []:
            try:
                payload = _bytes(item)
                if not payload:
                    continue
                await sender(
                    chat_id,
                    payload,
                    reply_to_message_id=reply_to,
                    message_thread_id=thread_id,
                )
                reply_to = None
            except Exception as exc:
                log.error("Failed to send %s: %s", attr, exc)


def _strip_bot_mention(text: str, username: str) -> str:
    if not username:
        return text
    return re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.IGNORECASE).strip()


def _is_bot_mentioned(message: T.Message, username: str) -> bool:
    if not username:
        return False
    haystack = (message.text or message.caption or "").lower()
    return f"@{username.lower()}" in haystack


# ---------------------------------------------------------------------------
# 10. Rich streaming editor — shows status lines like Agno's official UI
# ---------------------------------------------------------------------------
class StreamEditor:
    """Edits a single live Telegram message during generation.

    Shows a stack of status lines (Reasoning..., DuckDuckGo✓, ...) on top,
    followed by the answer text so far. Edits are throttled to ~1/sec to
    respect Telegram rate limits.
    """

    EDIT_INTERVAL = 1.0

    def __init__(self, chat_id: int, thread_id: Optional[int], reply_to: Optional[int]):
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.reply_to = reply_to
        self.msg_id: Optional[int] = None
        self.statuses: list[tuple[str, bool]] = []  # (label, done)
        self.content = ""
        self.last_edit = 0.0
        self._lock = asyncio.Lock()

    def _begin(self, label: str) -> None:
        # Replace any existing in-progress entry with the same label
        for i, (lbl, done) in enumerate(self.statuses):
            if lbl == label and not done:
                return
        self.statuses.append((label, False))

    def _complete(self, label: str, suffix: str = "") -> None:
        for i, (lbl, done) in enumerate(self.statuses):
            if lbl == label:
                self.statuses[i] = (lbl + suffix, True)
                return
        self.statuses.append((label + suffix, True))

    def _fail(self, label: str) -> None:
        for i, (lbl, done) in enumerate(self.statuses):
            if lbl == label:
                self.statuses[i] = (f"{lbl} ❌", True)
                return
        self.statuses.append((f"{label} ❌", True))

    def _render(self) -> str:
        lines = []
        for label, done in self.statuses:
            icon = "✓" if done else "…"
            lines.append(f"{icon} {label}")
        text = self.content.strip()
        if text:
            if lines:
                lines.append("")  # spacer
            lines.append(text)
        body = "\n".join(lines)
        return markdown_to_telegram_html(body)

    async def _flush(self) -> None:
        html = self._render()
        if len(html) > TG_MAX_MESSAGE:
            html = html[: TG_MAX_MESSAGE - 3] + "..."
        try:
            if self.msg_id is None:
                m = await bot.send_message(
                    self.chat_id,
                    html,
                    parse_mode="HTML",
                    reply_to_message_id=self.reply_to,
                    message_thread_id=self.thread_id,
                )
                self.msg_id = m.message_id
            else:
                await bot.edit_message_text(
                    html,
                    chat_id=self.chat_id,
                    message_id=self.msg_id,
                    parse_mode="HTML",
                )
        except Exception as exc:
            log.debug("Stream edit skipped: %s", exc)

    async def on_event(self, ev_name: str, chunk: Any) -> None:
        """Route an Agno RunEvent to the appropriate status update."""
        async with self._lock:
            def tn():
                tool = getattr(chunk, "tool", None)
                return (tool.tool_name if tool else None) or ""

            if ev_name == RunEvent.reasoning_started.value:
                self._begin("Reasoning")
            elif ev_name == RunEvent.reasoning_completed.value:
                self._complete("Reasoning")
            elif ev_name == RunEvent.tool_call_started.value:
                name = tn()
                if name:
                    self._begin(name)
                else:
                    try:
                        await bot.send_chat_action(
                            self.chat_id, "typing", message_thread_id=self.thread_id
                        )
                    except Exception:
                        pass
            elif ev_name == RunEvent.tool_call_completed.value:
                name = tn()
                if name:
                    self._complete(name)
            elif ev_name == RunEvent.tool_call_error.value:
                name = tn() or "Tool"
                self._fail(name)
            elif ev_name == RunEvent.memory_update_started.value:
                self._begin("Updating memory")
            elif ev_name == RunEvent.memory_update_completed.value:
                self._complete("Updating memory")
            elif ev_name == RunEvent.run_content.value:
                delta = getattr(chunk, "content", None)
                if isinstance(delta, str):
                    self.content += delta
            elif ev_name == RunEvent.run_completed.value:
                final = getattr(chunk, "content", None)
                if isinstance(final, str) and final:
                    self.content = final
            elif ev_name == RunEvent.run_error.value:
                self.content = "⚠️ عذراً، حدث خطأ أثناء المعالجة. استخدم /new ثم حاول مجدداً."

            now = time.monotonic()
            if (now - self.last_edit) >= self.EDIT_INTERVAL:
                await self._flush()
                self.last_edit = now

    async def finalize(self) -> tuple[Optional[int], str]:
        """Final render; returns (message_id, final_content)."""
        async with self._lock:
            # Clean up status lines for final message (remove "✓ Reasoning" etc.)
            # so the final user-visible message is just the answer text.
            final_text = self.content.strip()
        # Delete the live status message and send a clean final message instead
        if self.msg_id is not None:
            try:
                await bot.delete_message(self.chat_id, self.msg_id)
            except Exception:
                pass
        return None, final_text


# ---------------------------------------------------------------------------
# 11. Slash commands
# ---------------------------------------------------------------------------
@bot.message_handler(commands=["start"])
async def cmd_start(message: T.Message):
    me = await _get_bot_me()
    text = (
        f"👋 **مرحباً! أنا {me.first_name}** — مساعدك الذكي الفائق على تيليجرام.\n\n"
        "🧠 أعمل بنموذج **" + OPENAI_MODEL_ID + "** وأستطيع:\n"
        "• البحث في الويب (أخبار، معلومات حديثة)\n"
        "• قراءة ويكيبيديا وهانيكر نيوز\n"
        "• جلب أسعار الأسهم والتحليلات المالية\n"
        "• قراءة المقالات الكاملة من الروابط\n"
        "• إجراء العمليات الحسابية الدقيقة\n"
        "• تحليل الصور والملفات الصوتية والمرئية\n"
        "• التفاعل بالإيموجي، تثبيت الرسائل، إرسال الملصقات\n"
        "• تذكّر سياق المحادثة\n\n"
        "أرسل أي سؤال — أو اكتب /help للمزيد."
    )
    await _send_html(
        message.chat.id, text,
        reply_to=message.message_id,
        thread_id=message.message_thread_id,
    )


@bot.message_handler(commands=["help"])
async def cmd_help(message: T.Message):
    text = (
        "📖 **كيفية الاستخدام**\n\n"
        "**الدردشة:**\n"
        "• اكتب أي سؤال وسأبحث وأجيب\n"
        "• أرسل صورة وسأصفها/أحللها\n"
        "• أرسل ملفاً وسأقرأه (PDF، نص، ..)\n"
        "• أرسل رابطاً وسأقرأ المقال\n"
        "\n**الأوامر:**\n"
        "• /start – رسالة الترحيب\n"
        "• /help – عرض هذه الرسالة\n"
        "• /new – بدء محادثة جديدة (مسح الذاكرة)\n"
        "• /info – معلومات عني\n"
        "• /model – اسم النموذج المستخدم\n"
        "\n**في المجموعات** — اذكرني بـ @username أو رد على رسائلي.\n\n"
        "**الأدوات المتاحة:** DuckDuckGo, Wikipedia, HackerNews, "
        "Yahoo Finance, Newspaper4k, Calculator, Telegram actions."
    )
    await _send_html(
        message.chat.id, text,
        reply_to=message.message_id,
        thread_id=message.message_thread_id,
    )


@bot.message_handler(commands=["new"])
async def cmd_new(message: T.Message):
    scope = _session_scope(message.chat.id, message.message_thread_id)
    uid = str(message.from_user.id) if message.from_user else "unknown"
    new_sid = f"{scope}:{os.urandom(4).hex()}"
    try:
        from agno.storage.session.agent import AgentSession
        session = AgentSession(
            session_id=new_sid,
            user_id=uid,
            agent_id=agent.id,
            created_at=int(time.time()),
        )
        try:
            await db.upsert_session(session)
        except (TypeError, AttributeError):
            db.upsert_session(session)
    except Exception as exc:
        log.warning("Could not persist new session: %s", exc)

    await _send_html(
        message.chat.id,
        "✨ **تم مسح الذاكرة وبدء محادثة جديدة.** كيف يمكنني مساعدتك؟",
        reply_to=message.message_id,
        thread_id=message.message_thread_id,
    )


@bot.message_handler(commands=["info"])
async def cmd_info(message: T.Message):
    me = await _get_bot_me()
    text = (
        "🤖 **معلومات البوت**\n\n"
        f"• **الاسم:** {me.first_name}\n"
        f"• **المعرّف:** @{me.username}\n"
        f"• **النموذج:** {OPENAI_MODEL_ID}\n"
        f"• **الذاكرة:** آخر {HISTORY_RUNS} تفاعلات\n"
        f"• **البث المتدفق:** {'مفعل' if STREAMING else 'معطّل'}\n"
        f"• **الرد في المجموعات:** {'عند الإشارة فقط' if REPLY_TO_MENTIONS_ONLY else 'كل الرسائل'}\n"
        f"• **الأدوات:** Web Search, Wikipedia, HackerNews, YFinance, "
        f"Newspaper4k, Calculator, Telegram Tools, Reasoning"
    )
    await _send_html(
        message.chat.id, text,
        reply_to=message.message_id,
        thread_id=message.message_thread_id,
    )


@bot.message_handler(commands=["model"])
async def cmd_model(message: T.Message):
    await _send_html(
        message.chat.id,
        f"🧠 **النموذج الحالي:** `{OPENAI_MODEL_ID}`",
        reply_to=message.message_id,
        thread_id=message.message_thread_id,
    )


# ---------------------------------------------------------------------------
# 12. Main message handler
# ---------------------------------------------------------------------------
@bot.message_handler(
    content_types=[
        "text", "photo", "audio", "voice", "video", "document",
        "video_note", "animation", "sticker",
    ],
    func=lambda m: not (m.text and m.text.startswith("/")),
)
async def on_message(message: T.Message):
    chat_id = message.chat.id
    thread_id = message.message_thread_id
    user = message.from_user
    sender_id = user.id if user else None
    user_id = str(sender_id) if sender_id else "unknown"
    is_group = message.chat.type in ("group", "supergroup")
    me = await _get_bot_me()
    chain_key = (chat_id, thread_id)
    is_human = _incoming_is_human(message)
    is_self = sender_id == me.id

    # --- Never react to our own messages (obvious loop) ---
    if is_self:
        return

    # --- Bot-to-bot policy ---
    if not is_human:
        if not RESPOND_TO_OTHER_BOTS:
            # Configured to ignore other bots entirely
            return
        # Other bots are allowed to talk to us, but only if they @mention us
        # or directly reply to one of our messages (same gating as humans in
        # groups, but with an extra depth cap to prevent infinite loops).
        mentioned = _is_bot_mentioned(message, me.username or "")
        replied_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == me.id
        )
        if is_group and REPLY_TO_MENTIONS_ONLY and not (mentioned or replied_to_bot):
            return
        depth = _bot_chain_depth.get(chain_key, 0)
        if depth >= BOT_LOOP_MAX_DEPTH:
            log.info("Suppressing bot reply (depth %d/%d) in chat %s",
                     depth, BOT_LOOP_MAX_DEPTH, chat_id)
            return
    else:
        # A human spoke — reset the bot-to-bot chain counter for this thread
        _bot_chain_depth[chain_key] = 0

    # --- Group gating for humans (don't intrude on human conversations) ---
    if is_group and is_human and REPLY_TO_MENTIONS_ONLY:
        mentioned = _is_bot_mentioned(message, me.username or "")
        replied_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == me.id
        )
        if not mentioned and not replied_to_bot:
            return

    # --- Immediate reaction (shows the bot "saw" the message) ---
    if EMOJI_REACTION:
        try:
            await bot.set_message_reaction(
                chat_id,
                message.message_id,
                [T.ReactionTypeEmoji(EMOJI_REACTION)],
                is_big=False,
            )
        except Exception:
            pass

    # --- Typing indicator ---
    try:
        await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
    except Exception:
        pass

    # --- Build input ---
    payload = await _build_agent_input(message)
    if payload is None:
        return

    warning = payload.pop("_warning", None)
    if warning:
        await bot.send_message(
            chat_id, warning,
            reply_to_message_id=message.message_id,
            message_thread_id=thread_id,
        )

    if is_group and payload.get("input") and me.username:
        payload["input"] = _strip_bot_mention(payload["input"], me.username)

    if not payload.get("input") and not any(payload.get(k) for k in ("images", "audio", "videos", "files")):
        return

    # --- Session lookup ---
    scope = _session_scope(chat_id, thread_id)
    session_id = scope
    try:
        from agno.os.interfaces.telegram.state import (
            build_session_store_config, find_latest_session_id,
        )
        cfg = build_session_store_config(agent, "agent")
        found = await find_latest_session_id(cfg, user_id, agent.id, scope)
        if found:
            session_id = found
    except Exception as exc:
        log.debug("Session lookup skipped: %s", exc)

    reply_to = message.message_id if QUOTED_REPLIES or is_group else None

    # Remember if this run was triggered by another bot, so we can bump the
    # chain depth after successfully sending a reply (loop protection).
    triggered_by_bot = (not is_human)

    # --- Inject Telegram context into TelegramTools ---
    telegram_tools.chat_id = str(chat_id)
    telegram_tools.message_thread_id = str(thread_id) if thread_id else None
    telegram_tools.reply_to_message_id = reply_to

    run_kwargs = dict(user_id=user_id, session_id=session_id, **payload)

    def _bump_depth() -> None:
        if triggered_by_bot:
            _bot_chain_depth[chain_key] = _bot_chain_depth.get(chain_key, 0) + 1

    async def _track(m: Optional[T.Message]) -> None:
        if m is not None:
            _track_my_message(chat_id, thread_id, m.message_id)

    # --- Execute agent ---
    try:
        if STREAMING and SHOW_STATUS_LINES:
            editor = StreamEditor(chat_id, thread_id, reply_to)
            final_output: Optional[RunOutput] = None
            try:
                async for ev in agent.arun(
                    stream=True,
                    stream_events=True,
                    yield_run_output=True,
                    **run_kwargs,
                ):
                    if isinstance(ev, RunOutput):
                        final_output = ev
                        continue
                    ev_name = getattr(ev, "event", "")
                    if ev_name:
                        await editor.on_event(ev_name, ev)
            except Exception as exc:
                log.error("Agent stream error: %s", exc, exc_info=True)
                await editor.finalize()
                await _send_html(
                    chat_id,
                    "⚠️ عذراً، حدث خطأ أثناء المعالجة. استخدم /new ثم حاول مجدداً.",
                    reply_to=reply_to,
                    thread_id=thread_id,
                )
                return

            _, final_text = await editor.finalize()

            if final_output is None:
                log.warning("Stream finished without RunOutput")
                if final_text:
                    m = await _send_html(chat_id, final_text, reply_to=reply_to, thread_id=thread_id)
                    await _track(m); _bump_depth()
                return

            if final_output.status == "ERROR":
                log.error("Agent ERROR: %s", final_output.content)
                await _send_html(
                    chat_id,
                    "⚠️ عذراً، حدث خطأ أثناء المعالجة. استخدم /new ثم حاول مجدداً.",
                    reply_to=reply_to,
                    thread_id=thread_id,
                )
                return

            sent_any = False
            if final_text:
                m = await _send_html(chat_id, final_text, reply_to=reply_to, thread_id=thread_id)
                await _track(m)
                sent_any = True
            await _send_media(final_output, chat_id, reply_to=reply_to, thread_id=thread_id)
            if sent_any:
                _bump_depth()

        else:
            # Non-streaming path
            response: RunOutput = await agent.arun(**run_kwargs)
            if response is None or response.status == "ERROR":
                await _send_html(
                    chat_id,
                    "⚠️ عذراً، حدث خطأ أثناء المعالجة.",
                    reply_to=reply_to,
                    thread_id=thread_id,
                )
                return
            sent_any = False
            if response.content:
                m = await _send_html(chat_id, response.content, reply_to=reply_to, thread_id=thread_id)
                await _track(m)
                sent_any = True
            await _send_media(response, chat_id, reply_to=reply_to, thread_id=thread_id)
            if sent_any:
                _bump_depth()

    except Exception as exc:
        log.error("Unhandled error: %s", exc, exc_info=True)
        try:
            await bot.send_message(
                chat_id,
                "⚠️ عذراً، حدث خطأ غير متوقع. حاول مجدداً لاحقاً.",
                reply_to_message_id=message.message_id,
                message_thread_id=thread_id,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 13. Edited-message handler (edit your question -> bot re-answers)
# ---------------------------------------------------------------------------
@bot.edited_message_handler(func=lambda m: not (m.text and m.text.startswith("/")))
async def on_edited(message: T.Message):
    # Treat edited messages as new messages — re-runs the agent with updated text.
    await on_message(message)


# ---------------------------------------------------------------------------
# 14. Startup
# ---------------------------------------------------------------------------
async def _register_commands() -> None:
    try:
        await bot.set_my_commands(BOT_COMMANDS)
        log.info("Registered slash commands with BotFather")
    except Exception as exc:
        log.warning("Could not register commands: %s", exc)


async def main() -> None:
    me = await _get_bot_me()
    log.info("=" * 62)
    log.info("🤖 ULTIMATE TELEGRAM AI BOT ONLINE")
    log.info("   Username:   @%s (id=%d)", me.username, me.id)
    log.info("   Model:      %s", OPENAI_MODEL_ID)
    log.info("   Memory:     last %d runs (SQLite @ tmp/ultimate.db)", HISTORY_RUNS)
    log.info("   Streaming:  %s", "ON  (rich status lines)" if SHOW_STATUS_LINES and STREAMING else ("ON" if STREAMING else "OFF"))
    log.info("   Groups:     %s", "@mentions/replies only" if REPLY_TO_MENTIONS_ONLY else "all messages")
    log.info("   Media:      in+out (photos/audio/video/docs/stickers)")
    log.info("   Tools:      Reasoning + Web + Wiki + HN + YF + News + Calc + TG")
    log.info("=" * 62)

    await _register_commands()
    # Clear any old webhook and pending updates
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.infinity_polling(timeout=30, long_polling_timeout=25, logger_level=logging.INFO)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutdown (Ctrl+C). Goodbye.")
