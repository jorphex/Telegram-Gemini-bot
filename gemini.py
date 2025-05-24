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
FLASH_SYSTEM_INSTRUCTIONS = """instructions here"""
PRO_SYSTEM_INSTRUCTIONS = """instructions here"""


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
    model_id="gemini-2.5-pro-preview-05-06",
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

def split_text(text, max_length=4050):
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
    """Processes the accumulated message after the debounce timer expires."""
    global conversation_threads, pending_content, debounce_timers, FLASH_CONFIG, DEFAULT_TOOLS

    logger.info(f"Debounce timer expired for chat_id: {chat_id}. Processing accumulated message.")

    user_content_parts = pending_content.get(chat_id)
    if not user_content_parts:
        logger.warning(f"process_final_message called for chat_id {chat_id}, but no pending content found.")
        async with debounce_lock:
            if chat_id in debounce_timers: del debounce_timers[chat_id]
            if chat_id in pending_content: del pending_content[chat_id]
        return

    convo_state = conversation_threads.get(chat_id)
    chat_instance = None
    current_config = None
    if not convo_state or not convo_state.chat:
        logger.info(f"No active conversation for chat_id: {chat_id}. Creating default ({FLASH_CONFIG.name}).")
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        stop_typing_event_setup = asyncio.Event(); typing_task_setup = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event_setup))
        default_chat_instance = None
        try:
            default_chat_instance = await asyncio.to_thread(create_gemini_chat, FLASH_CONFIG, DEFAULT_TOOLS)
        except Exception as create_e: logger.error(f"Error creating default Gemini chat ({FLASH_CONFIG.name}) in thread: {create_e}")
        finally: stop_typing_event_setup.set(); await typing_task_setup
        if default_chat_instance:
            current_config = FLASH_CONFIG; chat_instance = default_chat_instance
            conversation_threads[chat_id] = ConversationState(chat=chat_instance, config=current_config)
            logger.info(f"Default conversation created successfully for chat_id: {chat_id} using {current_config.name}")
        else:
            logger.error(f"Failed to create default chat instance for chat_id: {chat_id}")
            await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, I couldn't start a conversation right now.")
            async with debounce_lock:
                if chat_id in debounce_timers: del debounce_timers[chat_id]
                if chat_id in pending_content: del pending_content[chat_id]
            return
    else:
        chat_instance = convo_state.chat
        current_config = convo_state.config

    log_display_parts = []
    if user_content_parts:
        for part in user_content_parts:
            if part is None: log_display_parts.append("[None Part]"); continue
            if hasattr(part, 'text') and part.text is not None:
                try: text_snippet = part.text[:100]; ellipsis = '...' if len(part.text) > 100 else ''; log_display_parts.append(f"[Text]: {text_snippet}{ellipsis}")
                except Exception as log_e: logger.error(f"Error logging text part content: {log_e}"); log_display_parts.append("[Text: Error logging content]")
            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                try: mime_type = getattr(part.inline_data, 'mime_type', 'unknown'); data_len = len(getattr(part.inline_data, 'data', b'')); log_display_parts.append(f"[Inline Data]: mime_type={mime_type}, size={data_len} bytes")
                except Exception as log_e: logger.error(f"Error logging inline_data part content: {log_e}"); log_display_parts.append("[Inline Data: Error logging content]")
            elif hasattr(part, 'file_data') and part.file_data is not None:
                try: mime_type = getattr(part.file_data, 'mime_type', 'unknown'); uri = getattr(part.file_data, 'file_uri', 'unknown'); log_display_parts.append(f"[File Data]: mime_type={mime_type}, uri={uri}")
                except Exception as log_e: logger.error(f"Error logging file_data part content: {log_e}"); log_display_parts.append("[File Data: Error logging content]")
            else: log_display_parts.append("[Unknown or Empty Part Structure]")
    else: log_display_parts.append("[No Parts Found]")
    user_content_log_display = " | ".join(log_display_parts)
    logger.info(f"Final combined message for chat_id: {chat_id} (using model: {current_config.name}): {user_content_log_display}")


    # --- Persistent Typing Indicator ---
    stop_typing_event = asyncio.Event()
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    typing_task = asyncio.create_task(keep_typing(context, chat_id, stop_typing_event))

    image_to_send: Optional[Dict[str, Any]] = None
    gemini_response_parts = []

    try:
        # --- Main Processing ---
        updated_chat = None
        response_object_or_error = None
        try:
            response_object_or_error, updated_chat = await asyncio.to_thread(
                query_assistant, user_content_parts, chat_instance, current_config
            )
            if updated_chat: conversation_threads[chat_id].chat = updated_chat
            else: logger.error(f"query_assistant returned None chat object for chat_id {chat_id}.")
        except Exception as e:
            logger.error(f"Error during query_assistant thread execution for chat_id {chat_id}: {e}")
            response_object_or_error = f"Error: Exception during AI query ({type(e).__name__})."

        # --- Extract Parts from Response Object or Error String ---
        if isinstance(response_object_or_error, str):
            logger.warning(f"Received error string from query_assistant: {response_object_or_error}")
            gemini_response_parts = [types.Part(text=response_object_or_error)]
        elif response_object_or_error and hasattr(response_object_or_error, 'candidates') and response_object_or_error.candidates:
            logger.info("Extracting parts from Gemini response object.")
            try:
                gemini_response_parts = response_object_or_error.candidates[0].content.parts
                logger.info(f"Found {len(gemini_response_parts)} parts in the response.")
            except (AttributeError, IndexError, TypeError) as e:
                 logger.error(f"Could not extract parts from response object: {e}")
                 gemini_response_parts = [types.Part(text="Error: Could not read AI response structure.")]
        else:
            logger.error(f"Invalid response object or error received from query_assistant: {type(response_object_or_error)}")
            gemini_response_parts = [types.Part(text="Error: Received invalid response from AI.")]

        # --- Process and Send Each Part Individually ---
        sent_anything_successfully = False
        parts_to_send_to_user = []
        last_part_was_search_tool_call = False

        for part_data in gemini_response_parts:
            is_search_tool_call = False
            if hasattr(part_data, 'executable_code') and part_data.executable_code and hasattr(part_data.executable_code, 'code'):
                code = part_data.executable_code.code.strip()
                if "search(" in code or "google_search(" in code or "concise_search(" in code:
                    logger.info(f"Detected search tool call, flagging for suppression: {code}")
                    is_search_tool_call = True
                    last_part_was_search_tool_call = True
                    continue

            if last_part_was_search_tool_call and hasattr(part_data, 'code_execution_result'):
                logger.info("Detected code execution result immediately after search tool call, flagging for suppression.")
                last_part_was_search_tool_call = False
                continue

            parts_to_send_to_user.append(part_data)
            last_part_was_search_tool_call = False

        total_display_parts = len(parts_to_send_to_user)
        original_text_segments_for_fallback = [""] * total_display_parts

        for part_idx, part_data in enumerate(parts_to_send_to_user):
            content_to_send = ""
            current_original_text = ""
            parse_mode_for_this_part = "MarkdownV2"

            if hasattr(part_data, 'text') and part_data.text:
                logger.info(f"Processing TEXT part {part_idx+1}/{total_display_parts}")
                current_original_text = part_data.text
                try:
                    content_to_send = telegramify_markdown.markdownify(part_data.text)
                except Exception as md_e:
                    logger.error(f"Error with telegramify.markdownify for text part: {md_e}. Falling back to plain text for this part.")
                    content_to_send = current_original_text
                    parse_mode_for_this_part = None

            elif hasattr(part_data, 'executable_code') and part_data.executable_code and hasattr(part_data.executable_code, 'code'):
                logger.info(f"Processing EXECUTABLE_CODE part {part_idx+1}/{total_display_parts}")
                code = part_data.executable_code.code.strip()
                current_original_text = code
                content_to_send = f"```python\n{code}\n```"

            elif hasattr(part_data, 'code_execution_result') and part_data.code_execution_result and hasattr(part_data.code_execution_result, 'output'):
                logger.info(f"Processing CODE_EXECUTION_RESULT part {part_idx+1}/{total_display_parts}")
                output = part_data.code_execution_result.output.strip()
                current_original_text = f"Output:\n{output}"
                content_to_send = f"**Output:**\n```\n{output}\n```"

            elif hasattr(part_data, 'inline_data') and part_data.inline_data and hasattr(part_data.inline_data, 'data'):
                mime_type = getattr(part_data.inline_data, 'mime_type', '')
                if mime_type.startswith('image/'):
                    img_data_bytes = getattr(part_data.inline_data, 'data', None)
                    if isinstance(img_data_bytes, bytes) and img_data_bytes:
                        image_to_send = {'data': img_data_bytes, 'mime_type': mime_type}
                        logger.info(f"Extracted image data ({mime_type}, {len(img_data_bytes)} bytes) to send separately after text parts.")
                        original_text_segments_for_fallback[part_idx] = "[Image will be sent separately]"
                        continue
                    else:
                        logger.warning("Found inline_data image part but data is missing or not bytes.")
                        content_to_send = "_[Error processing image data]_"
                        current_original_text = "[Error processing image data]"
                else:
                    logger.warning(f"Found inline_data part with non-image mime_type: {mime_type}. Skipping.")
                    original_text_segments_for_fallback[part_idx] = f"[Skipped non-image inline_data: {mime_type}]"
                    continue
            else:
                logger.warning(f"Skipping unknown or empty part {part_idx+1}/{total_display_parts}")
                original_text_segments_for_fallback[part_idx] = "[Skipped unknown/empty part]"
                continue

            original_text_segments_for_fallback[part_idx] = current_original_text

            chunks_to_send_for_this_part = split_text(content_to_send, 4050)
            original_chunks_for_this_part_fallback = split_text(current_original_text, 4050)

            for chunk_idx, text_chunk in enumerate(chunks_to_send_for_this_part):
                is_last_chunk_of_this_api_part = (chunk_idx == len(chunks_to_send_for_this_part) - 1)
                is_last_displayable_api_part = (part_idx == total_display_parts - 1)

                current_chunk_is_last_overall = is_last_displayable_api_part and \
                                                 is_last_chunk_of_this_api_part and \
                                                 not image_to_send
                reply_markup = get_new_conversation_keyboard() if current_chunk_is_last_overall else None

                final_text_chunk = text_chunk.strip()
                if not final_text_chunk:
                    logger.info(f"Skipping empty sub-chunk {part_idx+1}-{chunk_idx+1} after stripping.")
                    continue

                # Add (Part X of Y) indicator only if there are multiple *displayable* parts or sub-parts
                if total_display_parts > 1 or len(chunks_to_send_for_this_part) > 1:
                    part_indicator_text = f"{part_idx+1}"
                    if len(chunks_to_send_for_this_part) > 1:
                        part_indicator_text += f", sub-part {chunk_idx+1}/{len(chunks_to_send_for_this_part)}"
                    part_indicator_text += f" of {total_display_parts}" # Clarified indicator

                    escaped_indicator = f"\n\n_\\({part_indicator_text}\\)_" if parse_mode_for_this_part == "MarkdownV2" else f"\n\n({part_indicator_text})"
                    if len(final_text_chunk.encode('utf-8') if parse_mode_for_this_part == "MarkdownV2" else final_text_chunk) + \
                       len(escaped_indicator.encode('utf-8') if parse_mode_for_this_part == "MarkdownV2" else escaped_indicator) <= 4096:
                        final_text_chunk += escaped_indicator
                    else:
                        logger.warning("Chunk too long to add segment indicator.")

                try:
                    logger.info(f"Attempting to send segment {part_idx+1}-{chunk_idx+1} with parse_mode: {parse_mode_for_this_part}")
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=final_text_chunk,
                        parse_mode=parse_mode_for_this_part,
                        reply_markup=reply_markup,
                        disable_web_page_preview=True
                    )
                    sent_anything_successfully = True
                    logger.info(f"Sent segment {part_idx+1}-{chunk_idx+1}.")
                except BadRequest as e:
                    if "can't parse entities" in str(e).lower() or "unclosed" in str(e).lower() or "nested" in str(e).lower():
                        logger.warning(f"MarkdownV2 failed for segment {part_idx+1}-{chunk_idx+1}: {e}. Retrying as plain text.")
                        try:
                            plain_chunk_to_send = (original_chunks_for_this_part_fallback[chunk_idx]
                                                   if chunk_idx < len(original_chunks_for_this_part_fallback)
                                                   else text_chunk).strip()
                            if not plain_chunk_to_send:
                                logger.info(f"Skipping empty plain text fallback for segment {part_idx+1}-{chunk_idx+1}.")
                                continue

                            if total_display_parts > 1 or len(chunks_to_send_for_this_part) > 1:
                                part_indicator_text = f"Segment {part_idx+1}"
                                if len(chunks_to_send_for_this_part) > 1: part_indicator_text += f", sub-part {chunk_idx+1}/{len(chunks_to_send_for_this_part)}"
                                part_indicator_text += f" of {total_display_parts} displayable segments"
                                indicator = f"\n\n({part_indicator_text})"
                                if len(plain_chunk_to_send) + len(indicator) <= 4096: plain_chunk_to_send += indicator
                            if len(plain_chunk_to_send) > 4096: plain_chunk_to_send = plain_chunk_to_send[:4090] + "...]"
                            await context.bot.send_message(chat_id=chat_id, text=plain_chunk_to_send, reply_markup=reply_markup, disable_web_page_preview=True)
                            sent_anything_successfully = True
                            logger.info(f"Sent segment {part_idx+1}-{chunk_idx+1} as plain text fallback.")
                        except Exception as plain_e:
                            logger.error(f"Error sending segment {part_idx+1}-{chunk_idx+1} as plain text fallback: {plain_e}")
                    else:
                        logger.error(f"Error sending segment {part_idx+1}-{chunk_idx+1} (BadRequest, not parsing): {e}")
                except Exception as e:
                    logger.error(f"Error sending segment {part_idx+1}-{chunk_idx+1}: {e}")

                if len(chunks_to_send_for_this_part) > 1 and chunk_idx < len(chunks_to_send_for_this_part) -1 :
                    await asyncio.sleep(0.2)
            if total_display_parts > 1 and part_idx < total_display_parts - 1:
                await asyncio.sleep(0.3)

        # --- Send Extracted Image (if any) ---
        if image_to_send:
            logger.info(f"Attempting to send extracted image data.")
            image_reply_markup = get_new_conversation_keyboard()
            image_caption = image_to_send.get('caption', None)
            try:
                await context.bot.send_photo(chat_id=chat_id, photo=image_to_send['data'], caption=image_caption, reply_markup=image_reply_markup)
                logger.info(f"Successfully sent extracted image.")
                sent_anything_successfully = True
            except Exception as img_e:
                logger.error(f"Error sending extracted image: {img_e}")
                if not sent_anything_successfully:
                     try: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was an error sending the generated image.", reply_markup=get_new_conversation_keyboard())
                     except Exception: pass

        if not sent_anything_successfully:
             logger.error("Failed to send any part of the response.")
             try: await context.bot.send_message(chat_id=chat_id, text="😭 Sorry, there was an unexpected problem sending the response.", reply_markup=get_new_conversation_keyboard())
             except Exception: pass

    finally:
        # --- Stop the Typing Indicator & Cleanup ---
        stop_typing_event.set()
        await typing_task
        logger.debug(f"Typing indicator task finished for chat_id: {chat_id}")
        async with debounce_lock:
            if chat_id in debounce_timers and debounce_timers[chat_id].done(): del debounce_timers[chat_id]
            if chat_id in pending_content: del pending_content[chat_id]
        logger.info(f"Cleaned up debounce state for chat_id: {chat_id}")

            
# --- TELEGRAM ---
async def handle_message(update: Update, context: CallbackContext):
    """Handles incoming messages, accumulates them with debouncing, and triggers processing."""
    global conversation_threads, debounce_timers, pending_content, debounce_lock, MAX_DEBOUNCE_PARTS, FLASH_CONFIG

    if not update.message:
        logger.warning("Received an update without a message object. Skipping.")
        return

    chat_id = update.message.chat.id
    message_id = update.message.message_id

    # --- Process User Input into a list of Parts ---
    current_message_parts: List[types.Part] = []
    input_processed = False

    youtube_pattern = r'(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+(?:[?&]\S*)?)'

    try:
        if update.message.photo:
            photo = update.message.photo[-1]; photo_file = await photo.get_file()
            photo_bytes_io = BytesIO(); await photo_file.download_to_memory(photo_bytes_io); photo_bytes_io.seek(0)
            img_part = types.Part.from_bytes(data=photo_bytes_io.getvalue(), mime_type='image/jpeg')
            if update.message.caption: current_message_parts.append(types.Part(text=update.message.caption))
            current_message_parts.append(img_part)
            logger.info(f"Processing image (caption: {bool(update.message.caption)}) for chat_id: {chat_id}")
            input_processed = True

        elif update.message.text:
            message_text = update.message.text
            youtube_match = re.search(youtube_pattern, message_text)
            if youtube_match:
                youtube_url = youtube_match.group(1)
                logger.info(f"Detected YouTube URL: {youtube_url} in message from chat_id: {chat_id}")
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
            logger.info(f"Message {message_id} from chat_id: {chat_id} has no processable text or photo. Skipping.")
            return

    except Exception as e:
        logger.error(f"Error processing input for chat_id {chat_id}, message {message_id}: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Error: Unable to process your message input.", reply_to_message_id=message_id)
        return

    if not input_processed or not current_message_parts:
         logger.warning(f"Input processing finished for chat_id {chat_id}, message {message_id}, but resulted in no content parts. Skipping.")
         return

    # --- Debounce ---
    DEBOUNCE_DELAY = 1.0 # seconds

    async with debounce_lock:
        existing_task = debounce_timers.get(chat_id)
        accumulated_parts_count = 0

        if existing_task:
            logger.debug(f"Debounce timer active for chat_id: {chat_id}. Cancelling previous task.")
            existing_task.cancel()

            if chat_id in pending_content:
                if len(pending_content[chat_id]) >= MAX_DEBOUNCE_PARTS:
                    logger.warning(f"Max debounce parts ({MAX_DEBOUNCE_PARTS}) reached for chat_id {chat_id}. Ignoring latest message part(s).")
                    accumulated_parts_count = len(pending_content[chat_id])
                else:
                    can_merge = (
                        len(current_message_parts) == 1 and hasattr(current_message_parts[0], 'text') and
                        pending_content[chat_id] and hasattr(pending_content[chat_id][-1], 'text')
                    )

                    if can_merge:
                        last_text_part = pending_content[chat_id][-1]
                        new_text = current_message_parts[0].text
                        last_text_part.text = f"{last_text_part.text}\n{new_text}"
                        accumulated_parts_count = len(pending_content[chat_id])
                        logger.info(f"Merged text message into pending content for chat_id: {chat_id}. Total parts: {accumulated_parts_count}")
                    else:
                        pending_content[chat_id].extend(current_message_parts)
                        accumulated_parts_count = len(pending_content[chat_id])
                        logger.info(f"Appended message part(s) to pending content for chat_id: {chat_id}. Total parts: {accumulated_parts_count}")
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
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND) | filters.PHOTO & (~filters.COMMAND), handle_message))

    logger.info("Bot is running... Waiting for messages.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")

if __name__ == '__main__':
        start_bot()

