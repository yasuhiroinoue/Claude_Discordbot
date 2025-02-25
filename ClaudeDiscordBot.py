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

# The history is counted by the number of pairs of exchanges between the user and the assistant.
MAX_HISTORY = 2 * int(os.getenv("MAX_HISTORY", "0"))  # Default to 0 if not set

# The maximum number of characters per Discord message
MAX_DISCORD_LENGTH = 2000

# Initialize Anthropi API
anthropic = AsyncAnthropicVertex(region=GCP_REGION, project_id=GCP_PROJECT_ID)
LLM_MODEL = os.getenv("MODEL")
MAX_TOKEN = 16384
MAX_TOKEN_THINKING_BUDGET = 8192

# Initialize Discord bot
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

message_history = {}

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
        cleaned_text = clean_discord_message(message.content)
        
        # Check for !save command
        save_to_file = False
        if cleaned_text.startswith("!save "):
            save_to_file = True
            cleaned_text = cleaned_text.replace("!save ", "", 1)
        
        async with message.channel.typing():
            if message.attachments:
                await process_attachments(message, cleaned_text, save_to_file)
            else:
                await process_text_message(message, cleaned_text, save_to_file)

async def process_attachments(message, cleaned_text, save_to_file=False):
    # 画像やその他の添付ファイルを処理
    for attachment in message.attachments:
        file_extension = os.path.splitext(attachment.filename.lower())[1]
        ext_to_mime = {
            '.png': "image/png", 
            '.jpg': "image/jpeg", 
            '.jpeg': "image/jpeg", 
            '.gif': "image/gif", 
            '.webp': "image/webp",
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
        if file_extension in ext_to_mime:
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                await message.add_reaction('🎨')
                mime_type = ext_to_mime[file_extension]
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status != 200:
                            await message.channel.send('Unable to download the image.')
                            return
                        image_data = await resp.read()
                        resized_image_stream = resize_image_if_needed(image_data, file_extension)
                        resized_image_data = resized_image_stream.getvalue()
                        encoded_image_data = base64.b64encode(resized_image_data).decode("utf-8")
                        
                        # 画像付きユーザー発言を履歴に追加
                        update_message_history_with_image(message.author.id, cleaned_text, encoded_image_data, mime_type)
                        
                        # 会話履歴全体を取得してLLMに渡す
                        formatted_history = get_formatted_message_history(message.author.id)
                        response_text = await generate_response_with_image_and_text(formatted_history)
                        
                        # Add AI response to history
                        update_message_history(message.author.id, response_text, "assistant")
                        
                        if save_to_file:
                            await save_response_as_file(message, response_text)
                        else:
                            await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)
                        return
            else:
                await message.add_reaction('📄')
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status != 200:
                            await message.channel.send('Unable to download the text file.')
                            return
                        text_data = await resp.text()
                        combined_text = f"{cleaned_text}\n{text_data}" if cleaned_text else text_data
                        await process_text_message(message, combined_text, save_to_file)
                        return
        else:
            supported_extensions = ', '.join(ext_to_mime.keys())
            await message.channel.send(f"🗑️ Unsupported file extension. Supported extensions are: {supported_extensions}")

async def process_text_message(message, cleaned_text, save_to_file=False):
    # テキストメッセージの処理
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        message_history.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    update_message_history(message.author.id, cleaned_text, "user")
    formatted_history = get_formatted_message_history(message.author.id)
    response_text = await generate_response_with_text(formatted_history)
    update_message_history(message.author.id, response_text, "assistant")
    
    if save_to_file:
        await save_response_as_file(message, response_text)
    else:
        await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)

async def save_response_as_file(message, response_text):
    """
    Saves the response as a text file and sends it to the Discord channel.
    """
    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"claude_response_{timestamp}.md"
    
    # Create a file object with the response text
    file = discord.File(io.StringIO(response_text), filename=filename)
    
    # Send a message with the file attached
    await message.channel.send(f"💾 Here's your response as a file:", file=file)
    
    # Send a preview of first few lines (optional)
    preview_lines = response_text.split('\n')[:5]  # First 5 lines
    preview = '\n'.join(preview_lines)
    if len(preview_lines) >= 5:
        preview += "\n..."
    
    if preview.strip():  # Only send preview if there's content
        await message.channel.send(f"📝 Preview:\n```\n{preview}\n```")

def extract_response(answer):
    data = answer.model_dump()
    texts = [item.get("text") for item in data["content"] if item.get("type") == "text"]
    return "\n".join(texts)

async def generate_response_with_text(message_text):
    answer = await anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKEN,
        thinking={
            "type": "enabled",
            "budget_tokens": MAX_TOKEN_THINKING_BUDGET
        },
        messages=message_text,
    )
    return extract_response(answer)

async def generate_response_with_image_and_text(formatted_history):
    answer = await anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKEN,
        thinking={
            "type": "enabled",
            "budget_tokens": MAX_TOKEN_THINKING_BUDGET
        },
        messages=formatted_history,  # 履歴全体をそのまま渡す
    )
    return extract_response(answer)


def update_message_history(user_id, text, message_type):
    global message_history
    # テキストのみの場合はリストではなく文字列として保存
    new_message = {'role': message_type, 'content': text}
    if user_id in message_history:
        message_history[user_id].append(new_message)
        if message_type == 'assistant' and len(message_history[user_id]) > MAX_HISTORY:
            message_history[user_id].pop(0)
            if len(message_history[user_id]) > 0:
                message_history[user_id].pop(0)
    else:
        message_history[user_id] = [new_message]

def update_message_history_with_image(user_id, text, image_data, mime_type):
    global message_history
    # 画像＋テキストの複合コンテンツを作成
    content = [
        {"type": "text", "text": text if text else 'What is this a picture of?'},
        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_data}}
    ]
    new_message = {'role': 'user', 'content': content}
    if user_id in message_history:
        message_history[user_id].append(new_message)
        if len(message_history[user_id]) > MAX_HISTORY:
            message_history[user_id].pop(0)
            if len(message_history[user_id]) > 0:
                message_history[user_id].pop(0)
    else:
        message_history[user_id] = [new_message]

def get_formatted_message_history(user_id):
    # 履歴が存在しない場合のチェック
    if user_id not in message_history or not message_history[user_id]:
        return "No messages found for this user."
    
    # 履歴は既に画像・テキストの複合形式になっているのでそのまま返す
    formatted_messages = []
    for message in message_history[user_id]:
        role = message['role']
        content = message['content']
        formatted_messages.append({"role": role, "content": content})
    return formatted_messages

def clean_discord_message(input_string):
    bracket_pattern = re.compile(r'<[^>]+>')
    return bracket_pattern.sub('', input_string)


def resize_image_if_needed(image_bytes, file_extension, max_size_mb=1, step=10):
    format_map = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.gif': 'GIF', '.webp': 'WEBP'}
    img_format = format_map.get(file_extension.lower(), 'JPEG')
    img_stream = io.BytesIO(image_bytes)
    img = Image.open(img_stream)
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.ANTIALIAS  # 古いバージョンの Pillow 用

    while img_stream.getbuffer().nbytes > max_size_mb * 1024 * 1024:
        width, height = img.size
        img = img.resize(
            (int(width * (100 - step) / 100), int(height * (100 - step) / 100)),
            resample_filter
        )
        img_stream = io.BytesIO()
        img.save(img_stream, format=img_format)
    return img_stream


async def split_and_send_messages(message_system, text, max_length):
    """
    Splits the given text into chunks that respect word boundaries and sends them
    using the provided message system.
    """
    start = 0
    while start < len(text):
        if len(text) - start <= max_length:
            await message_system.channel.send(text[start:])
            break

        end = start + max_length
        while end > start and text[end-1] not in ' \n\r\t':
            end -= 1

        if end == start:
            end = start + max_length

        await message_system.channel.send(text[start:end].strip())
        start = end

# Run the bot
bot.run(DISCORD_BOT_TOKEN)
