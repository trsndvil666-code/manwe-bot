# -*- coding: utf-8 -*-
"""
مانوه سولیمو — ربات نقش‌آفرینی حرفه‌ای برای گروه‌های سرزمین میانه
"""

import os
import json
import random
import logging
import aiosqlite
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from groq import AsyncGroq
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction, ParseMode

from prompts import SYSTEM_PROMPT, RELATIONSHIP_HINTS

load_dotenv()

# -------------------- تنظیمات --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SPONTANEITY = float(os.getenv("SPONTANEITY", "0.32"))
ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

DB_PATH = "manwe_memory.db"
MAX_HISTORY = 12          # تعداد پیام‌های اخیر برای کانتکست
MAX_MEMORIES_PER_USER = 8

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("Manwe")

# -------------------- دیتابیس --------------------
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                user_id INTEGER PRIMARY KEY,
                character_name TEXT NOT NULL,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                memory TEXT NOT NULL,
                created_at TEXT,
                UNIQUE(user_id, memory)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                user_id INTEGER,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
        """)
        await db.commit()


async def set_role(user_id: int, character_name: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO roles (user_id, character_name, updated_at) VALUES (?, ?, ?)",
            (user_id, character_name.strip(), datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_role(user_id: int) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT character_name FROM roles WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def add_memory(user_id: int, memory: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO memories (user_id, memory, created_at) VALUES (?, ?, ?)",
            (user_id, memory.strip(), datetime.utcnow().isoformat()),
        )
        # نگه داشتن فقط آخرین حافظه‌ها
        await db.execute("""
            DELETE FROM memories
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            )
        """, (user_id, user_id, MAX_MEMORIES_PER_USER))
        await db.commit()


async def get_memories(user_id: int) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT memory FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, MAX_MEMORIES_PER_USER),
        ) as cursor:
            rows = await cursor.fetchall()
            return [r[0] for r in rows]


async def save_message(chat_id: int, user_id: int, role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_history (chat_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, user_id, role, content, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_recent_history(chat_id: int, limit: int = MAX_HISTORY) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT role, content FROM chat_history
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            # معکوس کردن برای ترتیب زمانی
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


# -------------------- هوش مصنوعی --------------------
client = AsyncGroq(api_key=GROQ_API_KEY)


async def generate_response(
    user_message: str,
    user_id: int,
    username: str,
    chat_id: int,
    is_reply_to_bot: bool = False,
) -> Optional[str]:
    character = await get_role(user_id)
    memories = await get_memories(user_id)
    history = await get_recent_history(chat_id)

    # ساخت پرامپت رابطه‌ای
    relation_hint = ""
    if character:
        key = character.lower().strip()
        relation_hint = RELATIONSHIP_HINTS.get(key) or RELATIONSHIP_HINTS.get(character) or (
            f"این فرد نقش «{character}» را بازی می‌کند. با توجه به شخصیت مانوه با او رفتار کن."
        )

    memory_text = ""
    if memories:
        memory_text = "حافظه‌های مهم قبلی درباره این فرد:\n- " + "\n- ".join(memories)

    system = SYSTEM_PROMPT
    if relation_hint:
        system += f"\n\n### اطلاعات فعلی درباره مخاطب:\n{relation_hint}"
    if memory_text:
        system += f"\n\n{memory_text}"

    messages = [{"role": "system", "content": system}]

    # اضافه کردن تاریخچه
    for h in history[-MAX_HISTORY:]:
        messages.append({"role": h["role"], "content": h["content"]})

    # پیام فعلی
    display_name = f"{character or username or 'ناشناس'}"
    messages.append({
        "role": "user",
        "content": f"[{display_name}]: {user_message}"
    })

    try:
        completion = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.85,
            max_tokens=450,
            top_p=0.9,
        )
        reply = completion.choices[0].message.content.strip()

        # ذخیره تاریخچه
        await save_message(chat_id, user_id, "user", f"[{display_name}]: {user_message}")
        await save_message(chat_id, 0, "assistant", reply)

        # گاهی یک حافظه کوتاه استخراج کن (ساده)
        if random.random() < 0.25 and len(user_message) > 20:
            short_mem = f"{display_name} گفت: {user_message[:120]}"
            await add_memory(user_id, short_mem)

        return reply
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return None


# -------------------- هندلرها --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "من مانوه سولیمو هستم.\n"
        "پادشاه والار و ارباب بادها.\n\n"
        "در گروه حضور دارم تا با اهالی آردا سخن بگویم.\n"
        "ادمین‌ها می‌توانند با دستور /setrole نقش افراد را تعیین کنند."
    )


async def setrole_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id not in ADMIN_IDS:
        await update.message.reply_text("این فرمان تنها در اختیار ادمین‌هاست.")
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "استفاده:\n/setrole <یوزرآیدی یا ریپلای> <نام شخصیت>\n"
            "مثال: /setrole 123456789 فئانور\n"
            "یا روی پیام فرد ریپلای کن و بنویس: /setrole فئانور"
        )
        return

    # اگر ریپلای باشد
    target_user_id = None
    character = None

    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
        character = " ".join(context.args)
    else:
        try:
            target_user_id = int(context.args[0])
            character = " ".join(context.args[1:])
        except ValueError:
            await update.message.reply_text("یوزرآیدی باید عدد باشد یا روی پیام فرد ریپلای کن.")
            return

    await set_role(target_user_id, character)
    await update.message.reply_text(
        f"نقش «{character}» برای کاربر {target_user_id} ثبت شد.\n"
        f"از این پس مانوه او را با این هویت می‌شناسد."
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    role = await get_role(update.effective_user.id)
    if role:
        await update.message.reply_text(f"تو در چشم من به‌عنوان «{role}» شناخته می‌شوی.")
    else:
        await update.message.reply_text("هنوز نقشی برای تو ثبت نشده است.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message: Message = update.message
    if not message or not message.text:
        return

    # نادیده گرفتن دستورات
    if message.text.startswith("/"):
        return

    user = message.from_user
    chat_id = message.chat_id
    text = message.text.strip()

    # تشخیص اینکه آیا باید جواب بدهیم
    bot_username = (await context.bot.get_me()).username.lower()
    is_mentioned = f"@{bot_username}" in text.lower()
    is_reply_to_bot = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.is_bot
        and message.reply_to_message.from_user.id == context.bot.id
    )

    should_reply = is_mentioned or is_reply_to_bot

    # خودجوش بودن متوسط
    if not should_reply:
        if random.random() > SPONTANEITY:
            return
        # در پیام‌های خیلی کوتاه کمتر جواب بده
        if len(text) < 8 and random.random() > 0.4:
            return

    # نشان دادن در حال تایپ
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    reply = await generate_response(
        user_message=text,
        user_id=user.id,
        username=user.username or user.first_name or "ناشناس",
        chat_id=chat_id,
        is_reply_to_bot=is_reply_to_bot,
    )

    if reply:
        await message.reply_text(reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Exception while handling an update:", exc_info=context.error)


# -------------------- اجرا --------------------
def main():
    if not TELEGRAM_TOKEN or not GROQ_API_KEY:
        raise RuntimeError("TELEGRAM_BOT_TOKEN و GROQ_API_KEY باید تنظیم شوند.")

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("setrole", setrole_command))
    application.add_handler(CommandHandler("whoami", whoami))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_error_handler(error_handler)

    # مقداردهی اولیه دیتابیس
    async def post_init(app: Application):
        await init_db()
        logger.info("دیتابیس آماده شد. مانوه سولیمو بیدار است.")

    application.post_init = post_init

    logger.info("ربات در حال راه‌اندازی...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()