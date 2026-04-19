import os
import re
import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv
from anthropic import AsyncAnthropicVertex
from PIL import Image
import io
import base64
import datetime
import mimetypes

import logging

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GCP_REGION = os.getenv("GCP_REGION")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# Setting for logging user IDs (for debugging)
DEBUG_LOG_USER_IDS = os.getenv("DEBUG_LOG_USER_IDS", "False").lower() == "true"

# Configure logging for user interactions
log_formatter = logging.Formatter(
    "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
log_handler = logging.FileHandler("bot_usage.log", encoding="utf-8")
log_handler.setFormatter(log_formatter)

usage_logger = logging.getLogger("bot_usage")
usage_logger.setLevel(logging.INFO)
usage_logger.addHandler(log_handler)
# Avoid propagating to root logger
usage_logger.propagate = False

if DEBUG_LOG_USER_IDS:
    print("Debug mode: User ID logging is enabled")
    print("Logging user interactions with IDs to bot_usage.log")
else:
    print("Production mode: User ID logging is disabled")

# Constants
MAX_HISTORY = 2 * int(os.getenv("MAX_HISTORY", "0"))
MAX_DISCORD_LENGTH = 2000
LLM_MODEL = os.getenv("MODEL")
MAX_TOKEN = 64000
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
WEB_SEARCH_MAX_USES = int(os.getenv("WEB_SEARCH_MAX_USES", "0"))

MAX_IMAGE_SIZE_MB = 1

# Initialize
anthropic = AsyncAnthropicVertex(region=GCP_REGION, project_id=GCP_PROJECT_ID)
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
message_history = {}

# Additional text extensions that mimetypes might miss or strictly type
TEXT_EXTENSIONS = {
    '.md', '.csv', '.xml', '.html', '.ini', '.log', '.yaml', '.yml', '.conf', '.env',
    '.c', '.h', '.cpp', '.hpp', '.py', '.rs', '.js', '.ts', '.cs', '.php', '.rb',
    '.pl', '.pm', '.swift', '.r', '.R', '.go', '.kt', '.java', '.lua', '.sql', '.sh', '.bat'
}

def get_mime_type(filename):
    """Determine the MIME type of a file."""
    # Initialize mimetypes if not already done
    if not mimetypes.inited:
        mimetypes.init()
        
    ext = os.path.splitext(filename.lower())[1]
    mime_type, _ = mimetypes.guess_type(filename)
    
    # Trust mimetypes for common media types
    if mime_type:
        return mime_type
        
    # Fallback/Override for code and text files
    if ext in TEXT_EXTENSIONS or ext == '.txt':
        return 'text/plain'
        
    return None



async def send_long_message(message_system, text, max_length):
    """Splits and sends long messages."""
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        if end < len(text):
            while end > start and text[end - 1] not in ' \n\r\t':
                end -= 1
            if end == start:
                end = start + max_length
        await message_system.channel.send(text[start:end].strip())
        start = end


async def save_response_as_file_and_send(message, response_text, is_thinking=False):
    """Saves response as a file and sends it with a preview."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_type = "thinking" if is_thinking else "response"
    filename = f"claude_{file_type}_{timestamp}.md"
    file = discord.File(io.BytesIO(response_text.encode()), filename=filename)
    
    if is_thinking:
        await message.channel.send(f"💭 Here's my thinking process as a file:", file=file)
    else:
        await message.channel.send(f"💾 Here's my response as a file:", file=file)

    # The preview section is common
    preview_lines = response_text.split('\n')[:5]
    preview = '\n'.join(preview_lines)
    if len(preview_lines) >= 5:
        preview += "\n..."
    if preview.strip():
        preview_label = "Thinking preview:" if is_thinking else "Response preview:"
        await message.channel.send(f"📝 {preview_label}\n```\n{preview}\n```")


def clean_message(input_string):
    """Removes brackets from the input string."""
    return re.sub(r'<[^>]+>', '', input_string)


def resize_image(image_bytes, file_extension, max_size_mb=MAX_IMAGE_SIZE_MB, step=10):
    """Resizes the image to be under the max size."""
    format_map = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.gif': 'GIF', '.webp': 'WEBP'}
    img_format = format_map.get(file_extension.lower(), 'JPEG')
    img = Image.open(io.BytesIO(image_bytes))
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS

    img_stream = io.BytesIO()
    while True:
        img_stream.seek(0)
        img_stream.truncate(0)
        img.save(img_stream, format=img_format)
        if img_stream.getbuffer().nbytes <= max_size_mb * 1024 * 1024:
            break
        width, height = img.size
        img = img.resize(
            (int(width * (100 - step) / 100), int(height * (100 - step) / 100)),
            resample_filter
        )
    return img_stream


def update_history(user_id, content, role):
    """Updates message history with a new message."""
    new_message = {'role': role, 'content': content}
    if user_id in message_history:
        message_history[user_id].append(new_message)
        if role == 'assistant' and len(message_history[user_id]) > MAX_HISTORY:
            message_history[user_id].pop(0)
            if message_history[user_id]:
                message_history[user_id].pop(0)
    else:
        message_history[user_id] = [new_message]


def get_history(user_id):
    """Retrieves formatted message history."""
    return message_history.get(user_id, [])


async def generate_response(message_text):
    """Generates a response using the Anthropic API with streaming."""
    thinking_output = ""
    response_output = ""
    
    stream_kwargs = {
        "model": LLM_MODEL,
        "max_tokens": MAX_TOKEN,
        "thinking": {"type": "adaptive"},
        "messages": message_text,
    }
    if SYSTEM_PROMPT:
        stream_kwargs["system"] = SYSTEM_PROMPT
    if WEB_SEARCH_MAX_USES > 0:
        stream_kwargs["tools"] = [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": WEB_SEARCH_MAX_USES,
        }]

    # Receive responses using the Stream API
    async with anthropic.messages.stream(**stream_kwargs) as stream:
        async for event in stream:
            if event.type == "content_block_start":
                # Record block start (if needed for debugging purposes)
                pass
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    # Accumulate the thinking process
                    thinking_output += event.delta.thinking
                    #For debugging purposes, you can print the thinking process
                    # print(f"{event.delta.thinking}", end="", flush=True)
                elif event.delta.type == "text_delta":
                    # Accumulate the text response
                    response_output += event.delta.text
            elif event.type == "content_block_stop":
                # Record block end (if needed for debugging purposes)
                pass

        if WEB_SEARCH_MAX_USES > 0:
            final_message = await stream.get_final_message()
            seen_urls = {}
            for block in final_message.content:
                if getattr(block, "type", None) != "text":
                    continue
                for c in getattr(block, "citations", None) or []:
                    url = getattr(c, "url", None)
                    if url and url not in seen_urls:
                        seen_urls[url] = getattr(c, "title", None) or url
            if seen_urls:
                sources_md = "\n\n**📚 Sources:**\n" + "\n".join(
                    f"- [{title}]({url})" for url, title in seen_urls.items()
                )
                response_output += sources_md
    
    # If using the thinking process, you can log or save it here
    # Example of logging: logging.debug(f"Thinking process: {thinking_output[:100]}...")
    
    # Return the completed response
    return thinking_output, response_output


async def send_response(message, response_text, save_to_file, is_thinking=False):
    """Common logic for sending responses"""
    if save_to_file:
        await save_response_as_file_and_send(message, response_text, is_thinking)
    else:
        prefix = "💭 My thinking: " if is_thinking else ""
        await send_long_message(message, prefix + response_text, MAX_DISCORD_LENGTH)


async def download_attachment(attachment):
    """Common logic for downloading attachments"""
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            if resp.status != 200:
                return None
            return await resp.read()


async def process_message_with_attachments(message, attachments, cleaned_text, save_to_file):
    """Process a message with multiple attachments"""
    content = []
    
    # Add the text content first if it exists
    if cleaned_text:
        content.append({"type": "text", "text": cleaned_text})

    # Filter valid attachments; warn on invalid ones but continue with the rest
    valid_attachments = []
    for attachment in attachments:
        if get_mime_type(attachment.filename):
            valid_attachments.append(attachment)
        else:
            await message.channel.send(f"🗑️ Unsupported file type/extension, skipping: {attachment.filename}")

    # If nothing usable remains (no text and no valid attachments), stop
    if not valid_attachments and not cleaned_text:
        return

    # Processing loop
    for attachment in valid_attachments:
        file_extension = os.path.splitext(attachment.filename.lower())[1]
        mime_type = get_mime_type(attachment.filename)

        file_data = await download_attachment(attachment)
        if file_data is None:
            await message.channel.send(f'Unable to download the file, skipping: {attachment.filename}')
            continue

        if mime_type.startswith('image/'):
            await message.add_reaction('🎨')
            resized_image_stream = resize_image(file_data, file_extension)
            encoded_data = base64.b64encode(resized_image_stream.getvalue()).decode("utf-8")
            content.append({
                "type": "image", 
                "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}
            })
        elif mime_type == 'application/pdf':
            await message.add_reaction('📄')
            encoded_data = base64.b64encode(file_data).decode("utf-8")
            content.append({
                "type": "document", 
                "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}
            })
        else:
            # Text based files (caught by get_mime_type falling back to text/plain or known extensions)
            await message.add_reaction('📄')

            text_data = file_data.decode('utf-8', errors='replace')
            file_info = (
                f"\n## File Information\n"
                f"- Name: `{attachment.filename}`\n"
                f"- Size: {len(text_data)} characters\n"
                f"- Type: {mime_type}\n\n"
            )
            # Add as a text block
            content.append({
                "type": "text",
                "text": f"{file_info}{text_data}"
            })

    # If no content (e.g. empty message with no valid attachments), do nothing
    if not content:
        return

    # If the message only contains explicit text and no other content was added (no attachments processed),
    # process_text would have been better, but this functions works too if we ensure format is correct.
    # However, if content is just one text block, it's effectively the same as process_text logic 
    # but passed as list. content is list of blocks.
    
    # If content has no text block at start (cleaned_text was empty) but has images,
    # we might want to add a default prompt if it's just images?
    # Existing logic had: "What is this a picture of?"
    
    has_text = any(block['type'] == 'text' for block in content)
    if not has_text:
        # Check if we have images/pdfs
        has_media = any(block['type'] in ['image', 'document'] for block in content)
        if has_media:
             # Add default prompt as first block
             content.insert(0, {"type": "text", "text": "What is this content?"})

    update_history(message.author.id, content, 'user')
    formatted_history = get_history(message.author.id)
    thinking_text, response_text = await generate_response(formatted_history)
    update_history(message.author.id, response_text, "assistant")

    if save_to_file and thinking_text:
        await send_response(message, thinking_text, True, is_thinking=True)
    await send_response(message, response_text, save_to_file, is_thinking=False)


async def process_text(message, cleaned_text, save_to_file=False):
    """Processes text messages."""
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        message_history.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    update_history(message.author.id, cleaned_text, "user")
    formatted_history = get_history(message.author.id)
    thinking_text, response_text = await generate_response(formatted_history)
    update_history(message.author.id, response_text, "assistant")
    if save_to_file and thinking_text:
        await send_response(message, thinking_text, True, is_thinking=True)
    await send_response(message, response_text, save_to_file, is_thinking=False)


@bot.event
async def on_ready():
    print(f"Claude Bot Logged in as {bot.user}")


# コマンドとして!saveを実装
@bot.command(name='save')
async def save_command(ctx, *, content=None):
    """Save the response as a file."""
    if not content:
        await ctx.send("Please provide text after the !save command.")
        return
    
    cleaned_text = clean_message(content)
    async with ctx.typing():
        if ctx.message.attachments:
            await process_message_with_attachments(ctx.message, ctx.message.attachments, cleaned_text, True)
        else:
            await process_text(ctx.message, cleaned_text, True)



@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Log the interaction only for DMs and only if debugging is enabled
    if isinstance(message.channel, discord.DMChannel) and DEBUG_LOG_USER_IDS:
        channel_info = f"DM ({message.channel.id})"
        usage_logger.info(
            f"Processing message from User: {message.author.name} (ID: {message.author.id}) in {channel_info}"
        )

    if message.mention_everyone:
        await message.channel.send(f'{bot.user}です')
        return
    
    # ボットコマンドの処理
    await bot.process_commands(message)

    # メンションまたはDMの場合は応答（!で始まるコマンドを除く）
    if (bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel)) and not message.content.startswith('!'):
        cleaned_text = clean_message(message.content)
        async with message.channel.typing():
            if message.attachments:
                await process_message_with_attachments(message, message.attachments, cleaned_text, False)
            else:
                await process_text(message, cleaned_text, False)


if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)