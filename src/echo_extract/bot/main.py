"""Telegram bot entry point for EchoExtract."""

import asyncio
import logging

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

from echo_extract.core.config import settings
from echo_extract.core.logging_config import setup_logging

logger = logging.getLogger(__name__)

# A Router groups related handlers. Routers are attached to the Dispatcher.
router = Router()


@router.message(Command("start"))
async def handle_start(message: Message) -> None:
    """Respond to /start with a welcome message."""
    await message.answer(
        "Welcome to EchoExtract!\n\n"
        "Send me an audio file or voice message and I'll transcribe it."
    )


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    """Respond to /help with usage instructions."""
    await message.answer(
        "Commands:\n"
        "/start - Show the welcome message\n"
        "/help - Show this help text"
    )


@router.message()
async def handle_unknown(message: Message) -> None:
    """Catch-all handler for messages no other handler matched."""
    await message.answer("I don't understand that yet. Try /help.")


async def main() -> None:
    """Start the bot and begin polling for updates."""
    setup_logging()

    if not settings.telegram_token:
        raise SystemExit(
            "No Telegram token found. Set ECHO_TELEGRAM_TOKEN in your .env file."
        )

    bot = Bot(token=settings.telegram_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info("Bot is starting...")
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())