import logging
import asyncio
from io import BytesIO
import re 
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction 
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, CallbackContext, filters
from telegram.error import BadRequest, TelegramError 
import telegramify_markdown 
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch

TELEGRAM_TOKEN = 'BOT_TOKEN'
GEMINI_API_KEY = 'GEMINI_KEY'

MODEL_FLASH = "gemini-2.0-flash"
MODEL_PRO_EXP = "gemini-2.5-pro-exp-03-25"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

conversation_threads = {}

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    exit()

def get_new_conversation_keyboard():
    keyboard = [
        [
            InlineKeyboardButton(text=f"✨ New Flash 2.0", callback_data=f"new_{MODEL_FLASH}"),
            InlineKeyboardButton(text=f"✨ New Pro Exp 2.5 (5 RPM/100 RPD)", callback_data=f"new_{MODEL_PRO_EXP}")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def split_text(text, max_length=4050):
    """Splits text into chunks, respecting newlines."""
    if not isinstance(text, str): return []
    if len(text) <= max_length: return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        if end >= len(text):
            chunks.append(text[start:])
            break
        last_newline = text.rfind('\n', start, end)
        if last_newline != -1 and last_newline > start:
            chunks.append(text[start:last_newline])
            start = last_newline + 1
        else:
            chunks.append(text[start:end])
            start = end
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            for i in range(0, len(chunk), max_length): final_chunks.append(chunk[i:i+max_length])
        elif chunk: final_chunks.append(chunk)
    return final_chunks

def balance_markdown(chunk):
    if chunk.count("```") % 2 != 0: chunk += "\n```"
    in_code_block = False
    single_tick_balance = 0
    i = 0
    while i < len(chunk):
        if chunk.startswith("```", i): in_code_block = not in_code_block; i += 3
        elif chunk[i] == '`' and not in_code_block and (i == 0 or chunk[i-1] != '\\'): single_tick_balance += 1; i += 1
        else: i += 1
    if single_tick_balance % 2 != 0 and not (chunk.count("```") % 2 != 0): chunk += "`"
    if chunk.count('**') % 2 != 0: chunk += '**'
    star_count = 0; i = 0
    while i < len(chunk):
        if chunk.startswith('**', i): i += 2
        elif chunk[i] == '*' and (i == 0 or chunk[i-1] != '\\'): star_count += 1; i += 1
        else: i += 1
    if star_count % 2 != 0: chunk += '*'
    if chunk.count('__') % 2 != 0: chunk += '__'
    underscore_count = 0; i = 0
    while i < len(chunk):
        if chunk.startswith('__', i): i += 2
        elif chunk[i] == '_' and (i == 0 or chunk[i-1] != '\\'): underscore_count += 1; i += 1
        else: i += 1
    if underscore_count % 2 != 0: chunk += '_'
    return chunk

def create_gemini_chat(model_name: str):
    logger.info(f"Creating new chat with {model_name}")
    try:
        system_instruction_text = (
            "Instructions here"
        )
        google_search_tool = Tool(google_search=GoogleSearch())
        chat = client.chats.create(
            model=model_name,
            config=GenerateContentConfig(
                system_instruction=system_instruction_text,
                temperature=0.8,
                tools=[google_search_tool],
                response_modalities=["TEXT"]
            )
        )
        logger.info(f"Created chat instance for {model_name}.")
        return chat
    except Exception as e:
        logger.error(f"Error creating chat for {model_name}: {e}")
        return None

def query_assistant(user_content, chat, model_name: str):
    if not chat:
        logger.error("query_assistant called with no chat instance.")
        return "Error: No active conversation found.", None
    logger.info(f"Sending message to {model_name}")
    reply_text = "Error: Sorry, I could not get a response. Please try again."
    try:
        response = chat.send_message(user_content)
        if hasattr(response, 'text') and response.text:
            reply_text = response.text
            logger.info(f"Gemini response ({model_name}): {reply_text}")
        else:
            finish_reason_name = 'N/A'
            if response and response.candidates:
                 try:
                      if hasattr(response.candidates[0], 'finish_reason'):
                           finish_reason_name = response.candidates[0].finish_reason.name
                 except (AttributeError, IndexError): pass
            logger.warning(f"Gemini response received but has no text content (model: {model_name}). Finish Reason: {finish_reason_name}")
    except Exception as e:
        logger.error(f"Error calling Gemini API ({model_name}): {type(e).__name__} - {e}")
    return reply_text, chat

async def keep_typing(context: CallbackContext, chat_id: int, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

            await asyncio.sleep(4)
        except TelegramError as e:

            logger.warning(f"Failed to send typing action to {chat_id}: {e}")

            if isinstance(e, (BadRequest)) and ('chat not found' in str(e) or 'user is blocked' in str(e)):
                 logger.error(f"Stopping typing indicator for {chat_id} due to critical error: {e}")
                 break
            await asyncio.sleep(10) 
        except Exception as e:
            logger.error(f"Unexpected error in keep_typing for {chat_id}: {e}")
            await asyncio.sleep(10) 

async def handle_message(update: Update, context: CallbackContext):
    global conversation_threads

    if not update.message:
        logger.warning("Received an update without a message object. Skipping.")
        return

    chat_id = update.message.chat.id
    message_id = update.message.message_id

    convo_data = conversation_threads.get(chat_id)
    chat_instance = None
    model_name = None

    if not convo_data or not convo_data.get('chat'):
        logger.info(f"No active conversation for chat_id: {chat_id}. Creating default ({MODEL_FLASH}).")

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:

            default_chat_instance = await asyncio.to_thread(create_gemini_chat, MODEL_FLASH)
        except Exception as create_e:
            logger.error(f"Error creating default Gemini chat ({MODEL_FLASH}) in thread: {create_e}")
            default_chat_instance = None

        if default_chat_instance:
            model_name = MODEL_FLASH
            chat_instance = default_chat_instance
            conversation_threads[chat_id] = {'chat': chat_instance, 'model': model_name}
            logger.info(f"Default conversation created successfully for chat_id: {chat_id} using {model_name}")
        else:
            logger.error(f"Failed to create default chat instance for chat_id: {chat_id}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="😭 Sorry, I couldn't start a conversation right now. Please try sending your message again later.",
                reply_to_message_id=message_id
            )
            return 
    else:
        chat_instance = convo_data['chat']
        model_name = convo_data['model']

    user_content = None
    user_content_log_display = "[Image Received]"
    if update.message.photo:

        if "pro" not in model_name.lower(): logger.warning(f"Received image for non-Pro model ({model_name}). Vision capabilities might be limited.")
        images = []; photo = update.message.photo[-1]; photo_file = await photo.get_file()
        try:
            photo_bytes_io = BytesIO(); await photo_file.download_to_memory(photo_bytes_io); photo_bytes_io.seek(0)
            image = Image.open(photo_bytes_io);
            if image.mode != 'RGB': image = image.convert('RGB')
            images.append(image); logger.info(f"Processing image for chat_id: {chat_id}")
        except Exception as e:
            logger.error(f"Error processing photo for chat_id {chat_id}: {e}")
            await context.bot.send_message(chat_id=chat_id, text="Error: Unable to process the image.", reply_to_message_id=message_id)
            return
        prompt_text = update.message.caption if update.message.caption else "Describe this image."
        user_content = [prompt_text] + images; user_content_log_display = f"[Image Caption]: {prompt_text}"
    elif update.message.text:
        user_content = update.message.text; user_content_log_display = user_content
    else:
        logger.info(f"Message {message_id} from chat_id: {chat_id} has no text or photo content. Skipping."); return

    logger.info(f"Received message from chat_id: {chat_id} (using model: {model_name}): {user_content_log_display}")

    stop_typing_event = asyncio.Event()

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    typing_task = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event))

    reply_text = "Error: An unexpected error occurred." 
    try:

        updated_chat = None
        try:
            reply_text, updated_chat = await asyncio.to_thread(query_assistant, user_content, chat_instance, model_name)
            if updated_chat:
                conversation_threads[chat_id]['chat'] = updated_chat
            else:
                 logger.error(f"query_assistant returned None chat object for chat_id {chat_id}.")

        except Exception as e:
            logger.error(f"Error during query_assistant thread execution for chat_id {chat_id}: {e}")

        try:
            converted_reply = telegramify_markdown.markdownify(reply_text)
        except Exception as e:
            logger.error(f"Error converting response to MarkdownV2 for chat_id {chat_id}: {e}")
            converted_reply = telegramify_markdown.escape_markdown(reply_text)

        logger.info(f"Sending formatted response to chat_id: {chat_id}")

        chunks = split_text(converted_reply, 4050)
        total_chunks = len(chunks)

        if not chunks:
             logger.warning(f"No content chunks to send for chat_id {chat_id}. Original reply: '{reply_text[:100]}...'")

             try:
                 await context.bot.send_message(
                     chat_id=chat_id,
                     text="Received an empty response.",
                     reply_markup=get_new_conversation_keyboard()
                 )
             except Exception as send_err:
                  logger.error(f"Error sending 'empty response' message: {send_err}")

             total_chunks = 0 

        for idx, chunk in enumerate(chunks):
            balanced_chunk = balance_markdown(chunk)
            is_last_chunk = (idx == total_chunks - 1)
            footer_text = "" 

            if total_chunks > 1:
                chunk_indicator = f"\n\n_\\(Part {idx+1} of {total_chunks}\\)_"
                if len(balanced_chunk) + len(chunk_indicator) <= 4096:
                    balanced_chunk += chunk_indicator
                else:
                    logger.warning(f"Chunk {idx+1}/{total_chunks} too long to add indicator directly.")

            if is_last_chunk and model_name == MODEL_PRO_EXP:
                footer_text = "\n\n\\(Note: Pro Exp 2\\.5 has limits: 5 requests per minute, 100 requests per day\\.\\)"
                if len(balanced_chunk) + len(footer_text) <= 4096:
                    balanced_chunk += footer_text
                else:
                    logger.warning(f"Last chunk for Pro model too long to add footer directly for chat_id {chat_id}.")

            reply_markup = get_new_conversation_keyboard() if is_last_chunk else None

            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=balanced_chunk,
                    parse_mode="MarkdownV2",
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
                logger.info(f"Sent chunk {idx+1}/{total_chunks} to chat_id: {chat_id}")

            except BadRequest as e:
                logger.error(f"Error sending chunk {idx+1}/{total_chunks} to chat_id {chat_id} (BadRequest): {e}")
                if "can't parse entities" in str(e) or "unclosed" in str(e) or "nested" in str(e):
                    logger.info(f"Retrying chunk {idx+1}/{total_chunks} as plain text due to Markdown error.")
                    try:
                        original_chunks = split_text(reply_text, 4050)
                        plain_chunk = original_chunks[idx] if idx < len(original_chunks) else chunk
                        if total_chunks > 1: plain_chunk += f"\n\n(Part {idx+1} of {total_chunks})"

                        if is_last_chunk and model_name == MODEL_PRO_EXP:
                            plain_chunk += "\n\n(Note: Pro Exp model limits: ~5 RPM / 100 RPD)"
                        if len(plain_chunk) > 4096: plain_chunk = plain_chunk[:4090] + "...]"
                        await context.bot.send_message(chat_id=chat_id, text=plain_chunk, reply_markup=reply_markup, disable_web_page_preview=True)
                    except Exception as plain_e:
                        logger.error(f"Error sending chunk {idx+1}/{total_chunks} as plain text: {plain_e}")
                        if idx == 0: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was a problem formatting and sending my response.")
                else:
                     if idx == 0: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was an issue sending my response.")
            except Exception as e:
                logger.error(f"Error sending chunk {idx+1}/{total_chunks} to chat_id {chat_id}: {e}")
                if idx == 0:
                    try: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was an unexpected problem sending my response.")
                    except Exception as final_e: logger.error(f"Failed even to send the final error message to chat_id {chat_id}: {final_e}")

            if total_chunks > 1 and not is_last_chunk:
                await asyncio.sleep(0.5)

    finally:

        stop_typing_event.set()

        await typing_task
        logger.debug(f"Typing indicator task finished for chat_id: {chat_id}")

async def start_new_conversation_callback(update: Update, context: CallbackContext):
    global conversation_threads
    query = update.callback_query
    await query.answer("✨ New conversation started.") 

    chat_id = query.message.chat.id
    callback_data = query.data

    try:
        if not callback_data.startswith("new_"): raise ValueError("Callback data does not start with 'new_'")
        model_name = callback_data.split("new_", 1)[1]
        if model_name not in [MODEL_FLASH, MODEL_PRO_EXP]: raise ValueError(f"Invalid model name '{model_name}' in callback data")
    except (IndexError, ValueError) as e:
        logger.error(f"Invalid callback data received: {callback_data} - Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Error: Invalid selection received\\. Please try clicking the buttons again\\.", parse_mode="MarkdownV2")
        return

    stop_typing_event = asyncio.Event()
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    typing_task = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event))

    new_chat_instance = None
    try:

        new_chat_instance = await asyncio.to_thread(create_gemini_chat, model_name)

        if new_chat_instance:
            conversation_threads[chat_id] = {'chat': new_chat_instance, 'model': model_name}
            logger.info(f"Started new conversation with {model_name} for chat_id: {chat_id}")

            model_display_name = "Flash 2.0" if model_name == MODEL_FLASH else "Pro Exp 2.5 (5 RPM/100 RPD)" if model_name == MODEL_PRO_EXP else model_name
            start_message = telegramify_markdown.markdownify(f"✨ New {model_display_name} conversation started. How can I help you?")

            await context.bot.send_message(
                chat_id=chat_id,
                text=start_message,
                parse_mode="MarkdownV2",

            )
        else:
            error_message = telegramify_markdown.markdownify(f"😭 Sorry, I couldn't start a new conversation with `{model_name}` right now\\. Please try again later\\.")
            await context.bot.send_message(chat_id=chat_id, text=error_message, parse_mode="MarkdownV2", reply_markup=get_new_conversation_keyboard())

    except Exception as create_e:
        logger.error(f"Error during create_gemini_chat thread execution or sending confirmation for {model_name}: {create_e}")

        await context.bot.send_message(chat_id=chat_id, text="😭 An unexpected error occurred while switching models.", reply_markup=get_new_conversation_keyboard())
    finally:

        stop_typing_event.set()
        await typing_task
        logger.debug(f"Typing indicator task finished for chat_id {chat_id} during model switch.")

def start_bot():
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    pattern_flash = f"^new_{re.escape(MODEL_FLASH)}$"
    pattern_pro_exp = f"^new_{re.escape(MODEL_PRO_EXP)}$"
    application.add_handler(CallbackQueryHandler(start_new_conversation_callback, pattern=pattern_flash))
    application.add_handler(CallbackQueryHandler(start_new_conversation_callback, pattern=pattern_pro_exp))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND) | filters.PHOTO & (~filters.COMMAND), handle_message))

    logger.info("Bot is running... Waiting for messages.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")

if __name__ == '__main__':
        start_bot()
