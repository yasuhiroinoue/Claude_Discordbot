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

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GCP_REGION = os.getenv("GCP_REGION")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# Constants
MAX_HISTORY = 2 * int(os.getenv("MAX_HISTORY", "0"))
MAX_DISCORD_LENGTH = 2000
LLM_MODEL = os.getenv("MODEL")
MAX_TOKEN = 64000
MAX_TOKEN_THINKING_BUDGET = 32000
MAX_IMAGE_SIZE_MB = 1

# Initialize
anthropic = AsyncAnthropicVertex(region=GCP_REGION, project_id=GCP_PROJECT_ID)
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())
message_history = {}

ext_to_mime = {
    '.png': "image/png",
    '.jpg': "image/jpeg",
    '.jpeg': "image/jpeg",
    '.gif': "image/gif",
    '.webp': "image/webp",
    '.pdf': "application/pdf",
    '.txt': "text/plain",
    '.md': "text/markdown",
    '.csv': "text/csv",
    '.json': "application/json",
    '.xml': "application/xml",
    '.html': "text/html",
    '.ini': "text/plain",
    '.log': "text/plain",
    '.yaml': "text/yaml",
    '.yml': "text/yaml",
    '.c': "text/x-c",
    '.h': "text/x-c",
    '.cpp': "text/x-c++",
    '.hpp': "text/x-c++",
    '.py': "text/x-python",
    '.rs': "text/x-rust",
    '.js': "text/javascript",
    '.cs': "text/x-csharp",
    '.php': "text/x-php",
    '.rb': "text/x-ruby",
    '.pl': "text/x-perl",
    '.pm': "text/x-perl",
    '.swift': "text/x-swift",
    '.r': "text/x-r",
    '.R': "text/x-r",
    '.go': "text/x-go"
}


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
    file = discord.File(io.StringIO(response_text), filename=filename)
    
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
    
    # Receive responses using the Stream API
    async with anthropic.messages.stream(
        model=LLM_MODEL,
        max_tokens=MAX_TOKEN,
        thinking={"type": "enabled", "budget_tokens": MAX_TOKEN_THINKING_BUDGET},
        messages=message_text,
    ) as stream:
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


async def process_attachment(message, attachment, cleaned_text, save_to_file):
    """Process different types of attachments with a unified approach"""
    file_extension = os.path.splitext(attachment.filename.lower())[1]
    mime_type = ext_to_mime.get(file_extension)
    
    if not mime_type:
        supported_extensions = ', '.join(ext_to_mime.keys())
        await message.channel.send(f"🗑️ Unsupported file extension. Supported extensions are: {supported_extensions}")
        return
    
    file_data = await download_attachment(attachment)
    if file_data is None:
        await message.channel.send(f'Unable to download the file: {attachment.filename}')
        return
    
    # Type-specific processing
    if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        await message.add_reaction('🎨')
        resized_image_stream = resize_image(file_data, file_extension)
        encoded_data = base64.b64encode(resized_image_stream.getvalue()).decode("utf-8")
        content = [
            {"type": "text", "text": cleaned_text if cleaned_text else 'What is this a picture of?'},
            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}}
        ]
    elif file_extension in ['.pdf']:
        await message.add_reaction('📄')
        encoded_data = base64.b64encode(file_data).decode("utf-8")
        content = [
            {"type": "text", "text": cleaned_text if cleaned_text else 'Explain this pdf'},
            {"type": "document", "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}}
        ]
    else:
        await message.add_reaction('📄')
        text_data = file_data.decode('utf-8', errors='replace')
        file_info = (
            f"## File Information\n"
            f"- Name: `{attachment.filename}`\n"
            f"- Size: {len(text_data)} characters\n"
            f"- Type: {mime_type}\n\n"
        )
        combined_text = f"{cleaned_text}\n\n{file_info}{text_data}" if cleaned_text else f"{file_info}{text_data}"
        
        # combined_text = f"{cleaned_text}\n{text_data}" if cleaned_text else text_data
        await process_text(message, combined_text, save_to_file)
        return
    
    # Common processing for image and PDF files
    update_history(message.author.id, content, 'user')
    formatted_history = get_history(message.author.id)
    thinking_text, response_text = await generate_response(formatted_history)
    update_history(message.author.id, response_text, "assistant")


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
    await send_response(message, thinking_text, True, is_thinking=True)
    await send_response(message, response_text, save_to_file, is_thinking=False)


@bot.event
async def on_ready():
    print(f"Claude Bot Logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.mention_everyone:
        await message.channel.send(f'{bot.user}です')
        return
    
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        cleaned_text = clean_message(message.content)
        save_to_file = False
        if cleaned_text.startswith("!save "):
            save_to_file = True
            cleaned_text = cleaned_text.replace("!save ", "", 1)

        async with message.channel.typing():
            if message.attachments:
                for attachment in message.attachments:
                    await process_attachment(message, attachment, cleaned_text, save_to_file)
                    break  # Process only the first attachment
            else:
                await process_text(message, cleaned_text, save_to_file)


bot.run(DISCORD_BOT_TOKEN)
