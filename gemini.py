import logging
import asyncio
from io import BytesIO
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    CallbackContext,
    filters
)
from telegram.error import BadRequest, TelegramError
import telegramify_markdown
from telegramify_markdown.type import ContentTypes
from telegramify_markdown.interpreters import TextInterpreter, InterpreterChain
from PIL import Image
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch, ToolCodeExecution 

TELEGRAM_TOKEN = 'BOT_TOKEN'
GEMINI_API_KEY = 'GEMINI_KEY'

# --- LOGGER ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Model/Bot Configuration ---

# System Instructions
FLASH_SYSTEM_INSTRUCTIONS = """instructions"""

PRO_SYSTEM_INSTRUCTIONS = """instructions"""

@dataclass
class ModelConfig:
    name: str
    model_id: str
    system_instruction: str
    temperature: float
    emoji: str
    callback_data: str

# Configs/modes
FLASH_CONFIG = ModelConfig(
    name="2.5 Flash",
    model_id="gemini-2.5-flash-preview-05-20",
    system_instruction=FLASH_SYSTEM_INSTRUCTIONS,
    temperature=0.8,
    emoji="💭",
    callback_data="new_flash"
)

PRO_CONFIG = ModelConfig(
    name="2.5 Pro",
    model_id="gemini-2.5-pro-preview-06-05",
    system_instruction=PRO_SYSTEM_INSTRUCTIONS,
    temperature=0.6,
    emoji="💡",
    callback_data="new_pro"
)

# Tools
GOOGLE_SEARCH_TOOL = Tool(google_search=GoogleSearch())
CODE_EXECUTION_TOOL = Tool(code_execution=ToolCodeExecution())

DEFAULT_TOOLS = [
    GOOGLE_SEARCH_TOOL,
    CODE_EXECUTION_TOOL,
]

@dataclass
class ConversationState:
    chat: Any
    config: ModelConfig

# Global dictionary to track conversation state per Telegram chat.
conversation_threads: Dict[int, ConversationState] = {}
debounce_timers: Dict[int, asyncio.Task] = {}
pending_content: Dict[int, List[types.Part]] = {}
debounce_lock = asyncio.Lock()
MAX_DEBOUNCE_PARTS = 10

# Gemini
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    exit()

# --- HELPERS ---
def get_new_conversation_keyboard():
    """Returns the InlineKeyboardMarkup using config data for all modes."""
    keyboard = [
        [
            InlineKeyboardButton(text=f"{FLASH_CONFIG.emoji} New {FLASH_CONFIG.name}", callback_data=FLASH_CONFIG.callback_data),
            InlineKeyboardButton(text=f"{PRO_CONFIG.emoji} New {PRO_CONFIG.name}", callback_data=PRO_CONFIG.callback_data)
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def split_text(text, max_length=4000):
    """
    Splits text into chunks <= max_length.
    Prioritizes splitting at double newlines (\n\n), then single newlines (\n).
    Falls back to hard splitting if necessary.
    """
    if not isinstance(text, str) or len(text) <= max_length:
        return [text] if text else []
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + max_length, len(text))
        chunk = text[current_pos:end_pos]
        if end_pos == len(text): chunks.append(chunk); break
        split_pos = chunk.rfind('\n\n')
        if split_pos == -1 or split_pos == 0: split_pos = chunk.rfind('\n')
        if split_pos != -1 and split_pos > 0:
            split_char_count = 2 if text[current_pos + split_pos : current_pos + split_pos + 2] == '\n\n' else 1
            chunks.append(text[current_pos : current_pos + split_pos + split_char_count])
            current_pos += split_pos + split_char_count
        else: chunks.append(chunk); current_pos = end_pos
    return [c for c in chunks if c and not c.isspace()]

def create_gemini_chat(config: ModelConfig, tools: List[Tool]) -> Optional[Any]:
    """Creates a new Gemini chat instance using the provided configuration."""
    logger.info(f"Creating new Gemini chat with model: {config.name} ({config.model_id}), Tools: {[str(t) for t in tools]}")
    try:
        chat = client.chats.create(
            model=config.model_id,
            config=GenerateContentConfig(
                system_instruction=config.system_instruction,
                temperature=config.temperature,
                tools=tools,
                response_modalities=["TEXT"]
            )
        )
        logger.info(f"Successfully created Gemini chat instance for model {config.name}.")
        return chat
    except Exception as e:
        logger.error(f"Error creating Gemini chat for model {config.name}: {e}")
        return None

def query_assistant(user_content: List[types.Part], chat: Any, config: ModelConfig):
    """
    Sends a message list to Gemini.
    Returns the full response object on success, or an error string on failure.
    Also returns the potentially updated chat object.
    """
    if not chat:
        logger.error("query_assistant called with no chat instance.")
        return "Error: No active conversation found.", None

    logger.info(f"Sending message to Gemini model: {config.name}")
    response_object = None
    error_message = "Error: Sorry, I could not get a response. Please try again."

    try:
        response_object = chat.send_message(user_content)

        if not response_object or not hasattr(response_object, 'candidates') or not response_object.candidates:
             logger.warning(f"Gemini response object seems invalid or empty (model: {config.name}).")
             finish_reason_name = 'N/A'
             if response_object and hasattr(response_object, 'candidates') and response_object.candidates and hasattr(response_object.candidates[0], 'finish_reason'):
                  try: finish_reason_name = response_object.candidates[0].finish_reason.name
                  except: pass
             logger.warning(f"Finish Reason (if available): {finish_reason_name}")
             return error_message, chat

        logger.info(f"Gemini response received successfully ({config.name}).")
        return response_object, chat

    except Exception as e:
        logger.error(f"Error calling Gemini API ({config.name}): {type(e).__name__} - {e}")
        return f"Error: An exception occurred while contacting the AI ({type(e).__name__}).", chat

# --- Typing Indicator Helper ---
async def keep_typing(context: CallbackContext, chat_id: int, stop_event: asyncio.Event):
    """Sends typing action periodically until stop_event is set."""
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





async def process_final_message(chat_id: int, context: CallbackContext):
    """
    Processes the final accumulated message, sends it to Gemini,
    and handles the response, including text, code execution, generated files, and formatting.
    """
    global conversation_threads, pending_content, debounce_timers, FLASH_CONFIG, DEFAULT_TOOLS
    logger.info(f"Debounce timer expired for chat_id: {chat_id}. Processing message.")

    async with debounce_lock:
        user_content_parts = pending_content.pop(chat_id, None)
        if chat_id in debounce_timers:
            del debounce_timers[chat_id]

    if not user_content_parts:
        logger.warning(f"process_final_message called for chat_id {chat_id}, but no content found.")
        return

    # --- Ensure Conversation Exists ---
    convo_state = conversation_threads.get(chat_id)
    if not convo_state or not convo_state.chat:
        logger.info(f"No active conversation for {chat_id}. Creating default ({FLASH_CONFIG.name}).")
        stop_typing_event_setup = asyncio.Event()
        typing_task_setup = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event_setup))
        try:
            default_chat_instance = await asyncio.to_thread(create_gemini_chat, FLASH_CONFIG, DEFAULT_TOOLS)
            if default_chat_instance:
                convo_state = ConversationState(chat=default_chat_instance, config=FLASH_CONFIG)
                conversation_threads[chat_id] = convo_state
                logger.info(f"Default conversation created for {chat_id} using {FLASH_CONFIG.name}")
            else:
                raise Exception("Failed to create default chat instance.")
        except Exception as create_e:
            logger.error(f"Failed to create default chat for {chat_id}: {create_e}")
            await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, I couldn't start a conversation right now.")
            return
        finally:
            stop_typing_event_setup.set()
            await typing_task_setup
    
    chat_instance = convo_state.chat
    current_config = convo_state.config
    
    # --- Send to Gemini and Get Response ---
    stop_typing_event = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event))
    
    response_object = None
    error_message = ""
    try:
        logger.info(f"Sending {len(user_content_parts)} parts to Gemini for chat_id: {chat_id}")
        response_object, updated_chat = await asyncio.to_thread(
            query_assistant, user_content_parts, chat_instance, current_config
        )
        if updated_chat:
            conversation_threads[chat_id].chat = updated_chat
        if isinstance(response_object, str):
             error_message = response_object
             response_object = None

    except Exception as e:
        logger.error(f"Exception during Gemini query for {chat_id}: {e}")
        error_message = f"Error: An exception occurred while contacting the AI ({type(e).__name__})."
    finally:
        # Clean up user-uploaded files
        for part in user_content_parts:
            if hasattr(part, 'name') and isinstance(part.name, str) and part.name.startswith("files/"):
                try:
                    await asyncio.to_thread(client.files.delete, name=part.name)
                    logger.info(f"Deleted user file from File API: {part.name}")
                except Exception as delete_e:
                    logger.error(f"Failed to delete user file {part.name}: {delete_e}")
        stop_typing_event.set()
        await typing_task

    # --- Manually Process Response Parts ---
    full_response_text = ""
    files_to_send = []
    
    if error_message:
        full_response_text = error_message
    elif response_object and hasattr(response_object, 'candidates') and response_object.candidates:
        response_parts = response_object.candidates[0].content.parts
        logger.info(f"Received {len(response_parts)} parts from Gemini. Processing...")
        text_parts_for_markdown = []
        
        for i, part in enumerate(response_parts):
            if part.text:
                logger.info(f"Part {i+1}: Found text part.")
                text_parts_for_markdown.append(part.text)
            
            if part.executable_code:
                logger.info(f"Part {i+1}: Found executable_code part.")
                code = part.executable_code.code
                text_parts_for_markdown.append(f"```python\n{code}\n```")
            
            if part.code_execution_result:
                logger.info(f"Part {i+1}: Found code_execution_result part.")
                if hasattr(part.code_execution_result, 'output') and part.code_execution_result.output:
                    output = part.code_execution_result.output
                    text_parts_for_markdown.append(f"**Output:**\n```\n{output}\n```")
            
            if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                logger.info(f"Part {i+1}: Found inline_data part (a generated file).")
                file_data = part.inline_data
                file_name = f"output.{file_data.mime_type.split('/')[-1]}"
                files_to_send.append({'data': file_data.data, 'mime_type': file_data.mime_type, 'name': file_name})
                logger.info(f"Extracted generated file: {file_name} ({file_data.mime_type})")
        
        full_response_text = "\n\n".join(text_parts_for_markdown)
    else:
        full_response_text = "Error: Received an invalid or empty response from the AI."
        logger.error(f"Invalid response object received for chat_id {chat_id}: {response_object}")

    if not full_response_text.strip() and not files_to_send:
        logger.warning(f"Gemini returned an empty response for chat_id: {chat_id}")
        full_response_text = "I received your message but didn't have anything to add!"

    # --- Send Formatted Text Messages ---
    sent_text_successfully = False
    if full_response_text.strip():
        logger.info("Preparing to send formatted text to user.")
        try:
            text_only_interpreter = InterpreterChain([TextInterpreter()])
            message_boxes = await telegramify_markdown.telegramify(
                full_response_text,
                interpreters_use=text_only_interpreter,
                max_word_count=4050
            )
            total_boxes = len(message_boxes)
            logger.info(f"Telegramify split text into {total_boxes} message(s).")
            for i, box in enumerate(message_boxes):
                is_last_box = (i == total_boxes - 1) and not files_to_send
                reply_markup = get_new_conversation_keyboard() if is_last_box else None
                try:
                    await context.bot.send_message(chat_id, box.content, parse_mode="MarkdownV2", reply_markup=reply_markup, disable_web_page_preview=True)
                    sent_text_successfully = True
                except BadRequest as e:
                    if "can't parse entities" in str(e).lower():
                        logger.warning(f"MarkdownV2 failed for a chunk: {e}. Retrying as code block.")
                        fallback_text = f"```\n{box.content}\n```"
                        await context.bot.send_message(chat_id, fallback_text, parse_mode="MarkdownV2", reply_markup=reply_markup, disable_web_page_preview=True)
                        sent_text_successfully = True
                    else: raise e
                if not is_last_box: await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"Fatal error in telegramify processing or sending loop for {chat_id}: {e}")
            await context.bot.send_message(chat_id, "😭 A critical error occurred while formatting my response.", reply_markup=get_new_conversation_keyboard())

    # --- Send Extracted Files ---
    if files_to_send:
        total_files = len(files_to_send)
        logger.info(f"Found {total_files} generated file(s) to send.")
        for i, file_info in enumerate(files_to_send):
            is_last_file = (i == total_files - 1)
            reply_markup = get_new_conversation_keyboard() if is_last_file else None
            file_to_send_obj = BytesIO(file_info['data'])
            file_to_send_obj.name = file_info['name']
            try:
                mime_type = file_info['mime_type']
                caption = f"Generated file: `{file_info['name']}`"
                logger.info(f"Sending file {i+1}/{total_files}: {file_info['name']}")
                if mime_type.startswith('image/'):
                    await context.bot.send_photo(chat_id, photo=file_to_send_obj, caption=caption, parse_mode="MarkdownV2", reply_markup=reply_markup)
                else:
                    await context.bot.send_document(chat_id, document=file_to_send_obj, filename=file_info['name'], caption=caption, parse_mode="MarkdownV2", reply_markup=reply_markup)
            except Exception as e:
                logger.error(f"Failed to send generated file {file_info['name']} to {chat_id}: {e}")
                if not sent_text_successfully and i == 0:
                    await context.bot.send_message(chat_id, f"😭 Error sending generated file: {file_info['name']}", reply_markup=get_new_conversation_keyboard())
            if not is_last_file: await asyncio.sleep(0.3)
    else:
        logger.info("No generated files found in the response.")


            
# --- TELEGRAM ---
async def handle_message(update: Update, context: CallbackContext):
    """Handles incoming messages, including text, photos, documents (PDFs), and YouTube URLs."""
    global conversation_threads, debounce_timers, pending_content, debounce_lock, MAX_DEBOUNCE_PARTS, FLASH_CONFIG

    if not update.message:
        logger.warning("Received an update without a message object. Skipping.")
        return

    chat_id = update.message.chat.id
    message_id = update.message.message_id

    current_message_parts: List[types.Part] = []
    input_processed = False
    user_prompt_for_media = ""

    youtube_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+(?:[?&]\S*)?)'

    try:
        if update.message.photo:
            photo = update.message.photo[-1]; photo_file = await photo.get_file()
            photo_bytes_io = BytesIO(); await photo_file.download_to_memory(photo_bytes_io); photo_bytes_io.seek(0)
            img_part = types.Part.from_bytes(data=photo_bytes_io.getvalue(), mime_type='image/jpeg') # Assuming JPEG
            user_prompt_for_media = update.message.caption if update.message.caption else "Describe this image."
            current_message_parts.append(types.Part(text=user_prompt_for_media))
            current_message_parts.append(img_part)
            logger.info(f"Processing image (caption: {bool(update.message.caption)}) for chat_id: {chat_id}")
            input_processed = True

        elif update.message.document:
            doc = update.message.document
            if doc.mime_type == 'application/pdf':
                logger.info(f"Received PDF document: {doc.file_name} (Size: {doc.file_size} bytes)")
                pdf_file = await doc.get_file()
                pdf_bytes_io = BytesIO()
                await pdf_file.download_to_memory(pdf_bytes_io)
                pdf_bytes_io.seek(0)
                pdf_data = pdf_bytes_io.getvalue()

                # Threshold for using File API (e.g., 19MB to be safe for 20MB request limit)
                FILE_API_THRESHOLD = 19 * 1024 * 1024

                pdf_part = None
                if doc.file_size < FILE_API_THRESHOLD:
                    logger.info(f"PDF size ({doc.file_size} bytes) is under threshold. Using inline data.")
                    pdf_part = types.Part.from_bytes(data=pdf_data, mime_type='application/pdf')
                else:
                    logger.info(f"PDF size ({doc.file_size} bytes) exceeds threshold. Using File API.")
                    try:
                        uploaded_file = await asyncio.to_thread(
                            client.files.upload,
                            file=BytesIO(pdf_data),
                            config=types.FileConfig(mime_type='application/pdf', display_name=doc.file_name)
                        )
                        pdf_part = uploaded_file
                        logger.info(f"Successfully uploaded PDF to File API: {uploaded_file.name}")
                    except Exception as upload_e:
                        logger.error(f"Error uploading PDF to File API: {upload_e}")
                        await context.bot.send_message(chat_id=chat_id, text="Error: Could not process the large PDF via File API.", reply_to_message_id=message_id)
                        return

                if pdf_part:
                    user_prompt_for_media = update.message.caption if update.message.caption else "Summarize this document."
                    current_message_parts.append(types.Part(text=user_prompt_for_media))
                    current_message_parts.append(pdf_part)
                    input_processed = True
            else:
                logger.info(f"Received document of unsupported type: {doc.mime_type}. Skipping.")
                await context.bot.send_message(chat_id=chat_id, text=f"Sorry, I can only process PDF documents, not {doc.mime_type}.", reply_to_message_id=message_id)
                return


        elif update.message.text:
            message_text = update.message.text
            youtube_match = re.search(youtube_pattern, message_text)

            if youtube_match:
                youtube_url = youtube_match.group(1); logger.info(f"Detected YouTube URL: {youtube_url} in message from chat_id: {chat_id}")
                text_before_url = message_text[:youtube_match.start()].strip()
                if text_before_url: current_message_parts.append(types.Part(text=text_before_url))
                current_message_parts.append(types.Part(file_data=types.FileData(mime_type="video/youtube", file_uri=youtube_url)))
                text_after_url = message_text[youtube_match.end():].strip()
                if text_after_url: current_message_parts.append(types.Part(text=text_after_url))
                elif not text_before_url and not text_after_url: current_message_parts.append(types.Part(text="Summarize this video."))
                input_processed = True
            else:
                current_message_parts.append(types.Part(text=message_text))
                input_processed = True
        else:
            logger.info(f"Message {message_id} from chat_id: {chat_id} has no processable content. Skipping.")
            return

    except Exception as e:
        logger.error(f"Error processing input for chat_id {chat_id}, message {message_id}: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Error: Unable to process your message input.", reply_to_message_id=message_id)
        return

    if not input_processed or not current_message_parts:
         logger.warning(f"Input processing finished for chat_id {chat_id}, message {message_id}, but resulted in no content parts. Skipping.")
         return

    # --- Debounce Logic ---
    DEBOUNCE_DELAY = 0.5 # seconds

    async with debounce_lock:
        existing_task = debounce_timers.get(chat_id)
        accumulated_parts_count = 0

        if existing_task:
            logger.debug(f"Debounce timer active for chat_id: {chat_id}. Cancelling previous task.")
            existing_task.cancel()
            if chat_id in pending_content:
                if len(pending_content[chat_id]) + len(current_message_parts) > MAX_DEBOUNCE_PARTS:
                    logger.warning(f"Max debounce parts ({MAX_DEBOUNCE_PARTS}) would be exceeded for chat_id {chat_id}. Ignoring latest message part(s).")
                    accumulated_parts_count = len(pending_content[chat_id])
                else:
                    can_merge_text = (
                        len(current_message_parts) == 1 and hasattr(current_message_parts[0], 'text') and
                        not any(hasattr(p, 'inline_data') or hasattr(p, 'file_data') for p in current_message_parts) and
                        pending_content[chat_id] and
                        len(pending_content[chat_id]) > 0 and
                        hasattr(pending_content[chat_id][-1], 'text') and
                        not any(hasattr(p, 'inline_data') or hasattr(p, 'file_data') for p in [pending_content[chat_id][-1]])
                    )

                    if can_merge_text:
                        last_text_part = pending_content[chat_id][-1]
                        new_text = current_message_parts[0].text
                        if last_text_part.text == user_prompt_for_media and user_prompt_for_media:
                             last_text_part.text = f"{last_text_part.text} {new_text}"
                             logger.info(f"Appended text to existing media prompt for chat_id: {chat_id}.")
                        else:
                             last_text_part.text = f"{last_text_part.text}\n{new_text}"
                             logger.info(f"Merged text message into pending content for chat_id: {chat_id}.")
                        accumulated_parts_count = len(pending_content[chat_id])
                    else:
                        pending_content[chat_id].extend(current_message_parts)
                        accumulated_parts_count = len(pending_content[chat_id])
                        logger.info(f"Appended new part(s) (media or text) to pending content for chat_id: {chat_id}. Total parts: {accumulated_parts_count}")
            else:
                logger.warning(f"Debounce task existed for chat_id {chat_id}, but no pending content found. Starting fresh.")
                pending_content[chat_id] = current_message_parts
                accumulated_parts_count = len(pending_content[chat_id])
        else:
            logger.info(f"Starting new debounce timer ({DEBOUNCE_DELAY}s) for chat_id: {chat_id}.")
            pending_content[chat_id] = current_message_parts
            accumulated_parts_count = len(pending_content[chat_id])

        if accumulated_parts_count > 0:
            new_task = asyncio.create_task(schedule_processing(chat_id, context, DEBOUNCE_DELAY))
            debounce_timers[chat_id] = new_task

async def schedule_processing(chat_id: int, context: CallbackContext, delay: float):
    """Waits for the delay and then calls the processing function, handling cancellation."""
    try:
        await asyncio.sleep(delay)
        logger.debug(f"Debounce delay finished naturally for chat_id: {chat_id}. Proceeding to process.")
        await process_final_message(chat_id, context)
    except asyncio.CancelledError:
        logger.info(f"Debounce timer cancelled for chat_id: {chat_id}.")
    except Exception as e:
        logger.error(f"Unexpected error in schedule_processing for chat_id {chat_id}: {e}")
        async with debounce_lock:
            if chat_id in debounce_timers: del debounce_timers[chat_id]
            if chat_id in pending_content: del pending_content[chat_id]

async def start_new_conversation_callback(update: Update, context: CallbackContext):
    """Handles button presses for starting a new conversation using ModelConfig."""
    global conversation_threads, FLASH_CONFIG, PRO_CONFIG, DEFAULT_TOOLS

    query = update.callback_query
    await query.answer("✨ Starting new conversation...")

    chat_id = query.message.chat.id
    callback_data = query.data

    selected_config: Optional[ModelConfig] = None
    selected_tools: List[Tool] = DEFAULT_TOOLS

    if callback_data == FLASH_CONFIG.callback_data:
        selected_config = FLASH_CONFIG
    elif callback_data == PRO_CONFIG.callback_data:
        selected_config = PRO_CONFIG
    else:
        logger.error(f"Invalid callback data received: {callback_data}")
        await context.bot.send_message(
            chat_id=chat_id,
            text="Error: Invalid selection received\\. Please try clicking the buttons again\\.",
            parse_mode="MarkdownV2",
            reply_markup=get_new_conversation_keyboard()
        )
        return

    # --- Persistent Typing Indicator ---
    stop_typing_event = asyncio.Event()
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    typing_task = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event))

    new_chat_instance = None
    try:
        new_chat_instance = await asyncio.to_thread(create_gemini_chat, selected_config, selected_tools)

        if new_chat_instance:
            conversation_threads[chat_id] = ConversationState(chat=new_chat_instance, config=selected_config)
            logger.info(f"Started new conversation with {selected_config.name} for chat_id: {chat_id} (Tools: {[str(t) for t in selected_tools]})")
            start_message = telegramify_markdown.markdownify(f"{selected_config.emoji} New {selected_config.name} conversation started. How can I help you?")
            await context.bot.send_message(
                chat_id=chat_id,
                text=start_message,
                parse_mode="MarkdownV2",
                reply_markup=get_new_conversation_keyboard()
            )
        else:
            error_message = telegramify_markdown.markdownify(f"😭 Sorry, I couldn't start a new conversation with `{selected_config.name}` right now\\. Please try again later\\.")
            await context.bot.send_message(
                chat_id=chat_id,
                text=error_message,
                parse_mode="MarkdownV2",
                reply_markup=get_new_conversation_keyboard()
            )
    except Exception as create_e:
        logger.error(f"Error during create_gemini_chat thread execution or sending confirmation for {selected_config.name}: {create_e}")
        error_text = "😭 An unexpected error occurred while switching models."
        await context.bot.send_message(
            chat_id=chat_id,
            text=error_text,
            reply_markup=get_new_conversation_keyboard()
        )
    finally:
        stop_typing_event.set()
        await typing_task
        logger.debug(f"Typing indicator task finished for chat_id {chat_id} during model switch.")


def start_bot():
    """Initializes and runs the Telegram bot."""
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CallbackQueryHandler(start_new_conversation_callback, pattern=f"^{FLASH_CONFIG.callback_data}$"))
    application.add_handler(CallbackQueryHandler(start_new_conversation_callback, pattern=f"^{PRO_CONFIG.callback_data}$"))

    application.add_handler(MessageHandler(
        filters.TEXT & (~filters.COMMAND) |
        filters.PHOTO & (~filters.COMMAND) |
        filters.Document.PDF & (~filters.COMMAND),
        handle_message
    ))

    logger.info("Bot is running... Waiting for messages.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")

if __name__ == '__main__':
        start_bot()
