import logging
import asyncio
from io import BytesIO
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, CallbackContext, filters
import telegramify_markdown
from PIL import Image
from google import genai
from google.genai import types

# --- CONFIGURATION ---
TELEGRAM_TOKEN = 'BOT_TOKEN'
GEMINI_API_KEY = 'GEMINI_KEY'

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",  # Timestamps added here
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global dictionary to track Gemini chat objects per Telegram chat.
conversation_threads = {}

# Initialize Gemini client.
client = genai.Client(api_key=GEMINI_API_KEY)

def split_text(text, max_length=4050):
    """
    Splits text into a list of chunks, trying to break at newline boundaries.
    If a line is too long, it will be split in fixed-size chunks.
    """
    if len(text) <= max_length:
        return [text]
    lines = text.split("\n")
    parts = []
    current = ""
    for line in lines:
        # +1 accounts for the newline we add.
        if len(current) + len(line) + 1 > max_length:
            parts.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        parts.append(current)
    # Further split any part that is still too long.
    final_parts = []
    for part in parts:
        if len(part) > max_length:
            for i in range(0, len(part), max_length):
                final_parts.append(part[i:i+max_length])
        else:
            final_parts.append(part)
    return final_parts

def query_assistant(user_content, chat=None):
    """
    Sends a message to the Gemini chat API and returns the assistant's reply.
    If no existing chat conversation exists, a new one is created with system instructions
    and with Google Search enabled as a tool (using dynamic retrieval).
    """
    if chat is None:
        system_instruction = (
            "instructions here"
        )
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                tools=[types.Tool(
                    google_search=types.GoogleSearchRetrieval(
                        dynamic_retrieval_config=types.DynamicRetrievalConfig(dynamic_threshold=0.6)
                    )
                )],
                response_modalities=["TEXT"]
            )
        )
    try:
        response = chat.send_message(user_content)
        reply_text = response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        reply_text = "Error: Could not get response."
    return reply_text, chat

def balance_markdown(chunk):
    """
    Ensures that markdown code blocks and inline code markers are balanced in the chunk.
    If an opening marker is not closed, it appends a closing marker.
    """
    # Balance triple backticks (code blocks)
    triple_count = chunk.count("```")
    if triple_count % 2 != 0:
        chunk += "\n```"
    # For inline code, count single backticks outside code blocks.
    # This is a simplified approach.
    single_count = chunk.replace("```", "").count("`")
    if single_count % 2 != 0:
        chunk += "`"
    return chunk

async def handle_message(update: Update, context: CallbackContext):
    global conversation_threads

    chat_id = update.message.chat.id
    message_id = update.message.message_id

    # Skip system messages or notifications (like when the bot is added) in group chats
    if not (update.message.text or update.message.caption or update.message.photo):
        logger.info(f"Message {message_id} from chat_id: {chat_id} has no valid content. Skipping.")
        return

    chat_instance = conversation_threads.get(chat_id, None)

    # Process text and/or images.
    user_content = None
    if update.message.photo:
        images = []
        for photo in update.message.photo:
            photo_file = await photo.get_file()
            try:
                photo_bytes = await photo_file.download_as_bytearray()
            except Exception as e:
                logger.error(f"Error downloading photo: {e}")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Error: Unable to download the image.",
                    reply_to_message_id=message_id
                )
                return
            image = Image.open(BytesIO(photo_bytes))
            images.append(image)
        # Use the caption if available; otherwise, supply a default prompt.
        prompt_text = update.message.caption if update.message.caption else "Please describe this image."
        user_content = [prompt_text] + images
    else:
        user_content = update.message.text

    logger.info(f"Received message from chat_id: {chat_id} (message_id: {message_id}): {user_content}")

    assistant_task = asyncio.create_task(
        asyncio.to_thread(query_assistant, user_content, chat_instance)
    )
    await asyncio.sleep(1)

    stop_typing = asyncio.Event()
    typing_task = None
    if not assistant_task.done():
        async def keep_typing():
            while not stop_typing.is_set():
                try:
                    await context.bot.send_chat_action(chat_id, "typing")
                    logger.info(f"Sent typing action to {chat_id}")
                except Exception as e:
                    logger.error(f"Error sending chat action: {e}")
                await asyncio.sleep(4)
        typing_task = asyncio.create_task(keep_typing())

    try:
        reply_text, updated_chat = await assistant_task
    except Exception as e:
        reply_text = f"Error: An unexpected error occurred: {str(e)}"
        updated_chat = chat_instance
    finally:
        if typing_task:
            stop_typing.set()
            await typing_task

    conversation_threads[chat_id] = updated_chat

    if reply_text.startswith("Error:"):
        reply_text += ("\n\nPlease reply to the previous valid response to maintain the current conversation or send /new to start a new conversation.")

    try:
        converted_reply = telegramify_markdown.markdownify(reply_text)
    except Exception as e:
        logger.error(f"Error converting markdown: {e}")
        converted_reply = reply_text

    logger.info(f"Gemini response: {converted_reply}")

    # Split long messages into chunks.
    chunks = split_text(converted_reply, 4050)
    total_chunks = len(chunks)

    # Attach the inline keyboard only to the last chunk.
    for idx, chunk in enumerate(chunks):
        if total_chunks > 1:
            chunk += f"\n\n_\\(Part {idx+1} of {total_chunks}\\)_"
        chunk = balance_markdown(chunk)  # Ensure markdown entities are balanced.
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(text="✨💬 New", callback_data="new_conversation")]]) if idx == total_chunks - 1 else None
        try:
            # Remove reply_to_message_id to send as standalone messages in group chats.
            await context.bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode="MarkdownV2",
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            logger.info(f"Sent chunk {idx+1} of {total_chunks} to chat_id: {chat_id}")
        except Exception as e:
            logger.error(f"Error sending chunk {idx+1} of {total_chunks}: {e}")
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="😭 Sorry, there was a problem sending my response."
                )
            except Exception as e:
                logger.error(f"Additionally, error sending error feedback: {e}")

async def new_conversation(update: Update, context: CallbackContext):
    """
    Command handler to reset the conversation.
    """
    chat_id = update.message.chat.id
    conversation_threads[chat_id] = None  # Reset conversation
    # Process the new conversation text through telegramify_markdown to escape reserved characters.
    new_text = telegramify_markdown.markdownify("✨ New conversation started. How can I help you?")
    await context.bot.send_message(chat_id=chat_id, text=new_text, parse_mode="MarkdownV2")

async def new_conversation_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    chat_id = query.message.chat.id
    conversation_threads[chat_id] = None  # Reset conversation
    await query.answer("✨ New conversation started.")
    new_text = telegramify_markdown.markdownify("✨ New conversation started.")
    await context.bot.send_message(chat_id=chat_id, text=new_text, parse_mode="MarkdownV2")

def start_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    # Command handler for /new to reset conversation.
    application.add_handler(CommandHandler("new", new_conversation))
    # CallbackQuery handler for the inline "New" button.
    application.add_handler(CallbackQueryHandler(new_conversation_callback, pattern="^new_conversation$"))
    # Message handler for all non-command messages.
    application.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), handle_message))
    logger.info("Bot is running... Waiting for messages.")
    application.run_polling()

if __name__ == '__main__':
    start_bot()
