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
MAX_TOKEN = 16384
MAX_TOKEN_THINKING_BUDGET = 8192
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


async def save_response_as_file_and_send(message, response_text):
    """Saves response as a file and sends it with a preview."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"claude_response_{timestamp}.md"
    file = discord.File(io.StringIO(response_text), filename=filename)
    await message.channel.send(f"💾 Here's your response as a file:", file=file)

    preview_lines = response_text.split('\n')[:5]
    preview = '\n'.join(preview_lines)
    if len(preview_lines) >= 5:
        preview += "\n..."
    if preview.strip():
        await message.channel.send(f"📝 Preview:\n```\n{preview}\n```")


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
    """Generates a response using the Anthropic API."""
    answer = await anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKEN,
        thinking={"type": "enabled", "budget_tokens": MAX_TOKEN_THINKING_BUDGET},
        messages=message_text,
    )
    return "\n".join(item.get("text") for item in answer.model_dump()["content"] if item.get("type") == "text")


async def process_image_attachment(message, attachment, cleaned_text, save_to_file):
    """Processes image attachments."""
    file_extension = os.path.splitext(attachment.filename.lower())[1]
    mime_type = ext_to_mime[file_extension]
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            if resp.status != 200:
                await message.channel.send('Unable to download the image.')
                return
            image_data = await resp.read()
    resized_image_stream = resize_image(image_data, file_extension)
    encoded_image_data = base64.b64encode(resized_image_stream.getvalue()).decode("utf-8")

    content = [
        {"type": "text", "text": cleaned_text if cleaned_text else 'What is this a picture of?'},
        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded_image_data}}
    ]
    update_history(message.author.id, content, 'user')
    formatted_history = get_history(message.author.id)
    response_text = await generate_response(formatted_history)
    update_history(message.author.id, response_text, "assistant")

    if save_to_file:
        await save_response_as_file_and_send(message, response_text)
    else:
        await send_long_message(message, response_text, MAX_DISCORD_LENGTH)

async def process_pdf_attachment(message, attachment, cleaned_text, save_to_file):

    file_extension = os.path.splitext(attachment.filename.lower())[1]
    mime_type = ext_to_mime[file_extension]
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            if resp.status != 200:
                await message.channel.send('Unable to download the pdf.')
                return
            pdf_data = await resp.read()
    encoded_pdf_data = base64.b64encode(pdf_data).decode("utf-8")

    content = [
        {"type": "text", "text": cleaned_text if cleaned_text else 'Explain this pdf'},
        {"type": "document", "source": {"type": "base64", "media_type": mime_type, "data": encoded_pdf_data}}
    ]
    update_history(message.author.id, content, 'user')
    formatted_history = get_history(message.author.id)
    response_text = await generate_response(formatted_history)
    update_history(message.author.id, response_text, "assistant")

    if save_to_file:
        await save_response_as_file_and_send(message, response_text)
    else:
        await send_long_message(message, response_text, MAX_DISCORD_LENGTH)

async def process_other_attachment(message, attachment, cleaned_text, save_to_file):
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            if resp.status != 200:
                await message.channel.send('Unable to download the text file.')
                return
            text_data = await resp.text()

    combined_text = f"{cleaned_text}\n{text_data}" if cleaned_text else text_data
    await process_text(message, combined_text, save_to_file)

async def process_attachments(message, cleaned_text, save_to_file=False):
    """Handles message attachments."""
    for attachment in message.attachments:
        file_extension = os.path.splitext(attachment.filename.lower())[1]

        if file_extension not in ext_to_mime:
            supported_extensions = ', '.join(ext_to_mime.keys())
            await message.channel.send(f"🗑️ Unsupported file extension. Supported extensions are: {supported_extensions}")
            continue

        if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            await message.add_reaction('🎨')
            await process_image_attachment(message, attachment, cleaned_text, save_to_file)
            return
        elif file_extension in ['.pdf']:
            await message.add_reaction('📄')
            await process_pdf_attachment(message, attachment, cleaned_text, save_to_file)
            return
        else:
            await message.add_reaction('📄')
            await process_other_attachment(message, attachment, cleaned_text, save_to_file)
            return

async def process_text(message, cleaned_text, save_to_file=False):
    """Processes text messages."""
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        message_history.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    update_history(message.author.id, cleaned_text, "user")
    formatted_history = get_history(message.author.id)
    response_text = await generate_response(formatted_history)
    update_history(message.author.id, response_text, "assistant")

    if save_to_file:
        await save_response_as_file_and_send(message, response_text)
    else:
        await send_long_message(message, response_text, MAX_DISCORD_LENGTH)


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
                await process_attachments(message, cleaned_text, save_to_file)
            else:
                await process_text(message, cleaned_text, save_to_file)


bot.run(DISCORD_BOT_TOKEN)