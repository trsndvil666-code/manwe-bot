# -*- coding: utf-8 -*-
"""Manwe Sulimo — Telegram roleplay bot powered by Groq."""
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from dotenv import load_dotenv
from groq import AsyncGroq
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from prompts import SYSTEM_PROMPT, RELATIONSHIP_HINTS, lore_context

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SPONTANEITY = min(1.0, max(0.0, float(os.getenv("SPONTANEITY", "0.12"))))
TEMPERATURE = min(1.5, max(0.0, float(os.getenv("TEMPERATURE", "0.88"))))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "750"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "18"))
MAX_MEMORIES_PER_USER = int(os.getenv("MAX_MEMORIES_PER_USER", "12"))
DB_PATH = os.getenv("DB_PATH", "manwe_memory.db")
ADMIN_IDS = {int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()}
BOT_REPLY_ONLY_WHEN_ADDRESSED = os.getenv("BOT_REPLY_ONLY_WHEN_ADDRESSED", "false").lower() == "true"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("Manwe")
client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""CREATE TABLE IF NOT EXISTS roles (
            user_id INTEGER PRIMARY KEY, character_name TEXT NOT NULL, updated_at TEXT)""")
        await db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            memory TEXT NOT NULL, created_at TEXT, UNIQUE(user_id, memory))""")
        await db.execute("""CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
            user_id INTEGER, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT)""")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_history_chat ON chat_history(chat_id, id)")
        await db.commit()


async def set_role(user_id: int, character_name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO roles(user_id, character_name, updated_at) VALUES(?,?,?)",
            (user_id, character_name.strip(), utc_now()))
        await db.commit()


async def get_role(user_id: int) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT character_name FROM roles WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return row[0] if row else None


async def add_memory(user_id: int, memory: str):
    memory = memory.strip()[:500]
    if not memory:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR IGNORE INTO memories(user_id,memory,created_at) VALUES(?,?,?)",
                         (user_id, memory, utc_now()))
        await db.execute("""DELETE FROM memories WHERE user_id=? AND id NOT IN
            (SELECT id FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?)""",
                         (user_id, user_id, MAX_MEMORIES_PER_USER))
        await db.commit()


async def get_memories(user_id: int) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT memory FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?",
                              (user_id, MAX_MEMORIES_PER_USER)) as cur:
            return [r[0] for r in await cur.fetchall()]


async def save_message(chat_id: int, user_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO chat_history(chat_id,user_id,role,content,created_at) VALUES(?,?,?,?,?)",
                         (chat_id, user_id, role, content[:6000], utc_now()))
        # محدود نگه داشتن دیتابیس؛ پیام‌های تازه‌تر باقی می‌مانند.
        await db.execute("""DELETE FROM chat_history WHERE chat_id=? AND id NOT IN
            (SELECT id FROM chat_history WHERE chat_id=? ORDER BY id DESC LIMIT 160)""",
                         (chat_id, chat_id))
        await db.commit()


async def get_recent_history(chat_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""SELECT role,content FROM chat_history WHERE chat_id=?
            ORDER BY id DESC LIMIT ?""", (chat_id, MAX_HISTORY)) as cur:
            rows = await cur.fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows) if r in ("user", "assistant")]


async def generate_response(user_message: str, user_id: int, username: str,
                            chat_id: int, is_reply_to_bot: bool = False) -> Optional[str]:
    if client is None:
        logger.error("GROQ_API_KEY is not configured")
        return None

    character = await get_role(user_id)
    memories = await get_memories(user_id)
    history = await get_recent_history(chat_id)
    display_name = character or username or "یکی از اهالی آردا"

    relation = ""
    if character:
        key = character.casefold().strip()
        relation = RELATIONSHIP_HINTS.get(key, f"مخاطب نقش «{character}» را دارد؛ واکنش مانوه را با شناخت دقیق و بدون کلیشه‌سازی از این شخصیت تنظیم کن.")

    system = SYSTEM_PROMPT + "\n\n" + lore_context(user_message, character or "")
    if relation:
        system += "\n\nراهنمای رابطه با مخاطب:\n" + relation
    if memories:
        system += "\n\nچیزهایی که مانوه از گفت‌وگوهای قبلی این کاربر به خاطر دارد:\n- " + "\n- ".join(memories)

    # پیام جاری به‌صورت جداگانه برچسب‌گذاری می‌شود؛ تاریخچه خام باقی می‌ماند.
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"[گوینده: {display_name} | شناسه تلگرام: {user_id}]\n{user_message}"})

    try:
        result = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=0.92,
        )
        reply = (result.choices[0].message.content or "").strip()
        if not reply:
            return None
        await save_message(chat_id, user_id, "user", f"[{display_name}]: {user_message}")
        await save_message(chat_id, 0, "assistant", reply)
        # فقط نکته‌های شخصی روشن و نسبتاً پایدار را ذخیره می‌کنیم؛ نه هر پیام تصادفی.
        if len(user_message) > 45 and any(x in user_message for x in
                ("یادت باشه", "یادت باشد", "علاقه دارم", "دوست دارم", "از این به بعد")):
            await add_memory(user_id, f"{display_name}: {user_message[:220]}")
        return reply
    except Exception:
        logger.exception("Groq request failed")
        return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "آه، یکی دیگر از اهالی آردا! من مانوه‌ام؛ فرمانروای بادها و پادشاه والار.\n"
        "لازم نیست با من تشریفات به‌جا بیاوری. حرف بزن، سؤال بپرس یا مرا خطاب کن.\n"
        "ادمین‌ها می‌توانند با /setrole نقش افراد را تعیین کنند."
    )


async def setrole_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    actor = update.effective_user
    if not actor or actor.id not in ADMIN_IDS:
        await msg.reply_text("این فرمان در اختیار ادمین‌های تعیین‌شده است.")
        return

    if msg.reply_to_message:
        target = msg.reply_to_message.from_user
        character = " ".join(context.args).strip()
        if not character:
            await msg.reply_text("روی پیام شخص ریپلای کن و بعد بنویس: /setrole فئانور")
            return
    else:
        if len(context.args) < 2:
            await msg.reply_text("روش استفاده:\n/setrole شناسه_عددی نام_شخصیت\nیا روی پیام فرد ریپلای کن: /setrole فئانور")
            return
        try:
            target_id = int(context.args[0])
        except ValueError:
            await msg.reply_text("شناسه باید عددی باشد. برای روش ساده‌تر روی پیام فرد ریپلای کن.")
            return
        target = None
        character = " ".join(context.args[1:]).strip()
        await set_role(target_id, character)
        await msg.reply_text(f"بسیار خوب؛ از این پس شناسه {target_id} را با هویت «{character}» می‌شناسم.")
        return

    await set_role(target.id, character)
    await msg.reply_text(f"ثبت شد. از این پس {target.full_name} را «{character}» می‌شناسم.")


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    role = await get_role(update.effective_user.id)
    await update.effective_message.reply_text(
        f"در دفتر من با نام «{role}» ثبت شده‌ای." if role else
        "هنوز برایت نقشی تعیین نشده. شاید هم ترجیح می‌دهی خودت باشی؛ انتخاب بدی نیست.")


async def forgetme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM memories WHERE user_id=?", (uid,))
        await db.commit()
    await update.effective_message.reply_text("حافظهٔ شخصی‌ام دربارهٔ تو پاک شد.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    user = update.effective_user
    if not msg or not msg.text or not user or msg.text.startswith("/"):
        return
    text = msg.text.strip()
    bot = await context.bot.get_me()
    bot_username = (bot.username or "").casefold()
    mentioned = bool(bot_username and f"@{bot_username}" in text.casefold())
    replied = bool(msg.reply_to_message and msg.reply_to_message.from_user
                   and msg.reply_to_message.from_user.id == context.bot.id)
    addressed = mentioned or replied

    if BOT_REPLY_ONLY_WHEN_ADDRESSED and not addressed:
        return
    if not addressed:
        if random.random() > SPONTANEITY:
            return
        if len(text) < 10 and random.random() > 0.2:
            return

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)
    reply = await generate_response(text, user.id, user.username or user.first_name or "ناشناس",
                                    msg.chat_id, replied)
    if reply:
        await msg.reply_text(reply, disable_web_page_preview=True)
    else:
        await msg.reply_text("بادها فعلاً پیغام را با خود بردند. کمی بعد دوباره صدایم کن.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Telegram update failed", exc_info=context.error)


def main():
    if not TELEGRAM_TOKEN or not GROQ_API_KEY:
        raise RuntimeError("در Railway متغیرهای TELEGRAM_BOT_TOKEN و GROQ_API_KEY را تنظیم کن.")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setrole", setrole_command))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("forgetme", forgetme))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    async def post_init(application: Application):
        await init_db()
        logger.info("Manwe is ready; lore database loaded from prompts.py")

    app.post_init = post_init
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
