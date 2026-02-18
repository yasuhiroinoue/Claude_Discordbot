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
import json
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

MAX_IMAGE_SIZE_MB = 1

# Initialize
anthropic = AsyncAnthropicVertex(region=GCP_REGION, project_id=GCP_PROJECT_ID)
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
message_history = {}

# グラフィックレコーディング用のテンプレート
GRAPHIC_RECORDING_TEMPLATE = """
# グラフィックレコーディング風インフォグラフィック変換プロンプト
## 目的
  以下の内容を、超一流デザイナーが作成したような、日本語で完璧なグラフィックレコーディング風のHTMLインフォグラフィックに変換してください。情報設計とビジュアルデザインの両面で最高水準を目指します
  手書き風の図形やアイコンを活用して内容を視覚的に表現します。
## デザイン仕様
### 1. カラースキーム
```
  <palette>
  <color name='ファッション-1' rgb='593C47' r='89' g='59' b='70' />
  <color name='ファッション-2' rgb='F2E63D' r='242' g='230' b='60' />
  <color name='ファッション-3' rgb='F2C53D' r='242' g='196' b='60' />
  <color name='ファッション-4' rgb='F25C05' r='242' g='91' b='4' />
  <color name='ファッション-5' rgb='F24405' r='242' g='68' b='4' />
  </palette>
```
### 2. グラフィックレコーディング要素
- 左上から右へ、上から下へと情報を順次配置
- 日本語の手書き風フォントの使用（Yomogi, Zen Kurenaido, Kaisei Decol）
- 手描き風の囲み線、矢印、バナー、吹き出し
- テキストと視覚要素（アイコン、シンプルな図形）の組み合わせ
- キーワードの強調（色付き下線、マーカー効果）
- 関連する概念を線や矢印で接続
- 絵文字やアイコンを効果的に配置（✏️📌📝🔍📊など）
### 3. タイポグラフィ
  - タイトル：32px、グラデーション効果、太字
  - サブタイトル：16px、#475569
  - セクション見出し：18px、#1e40af、アイコン付き
  - 本文：14px、#334155、行間1.4
  - フォント指定：
    ```html
    <style>
    
@import
 url('https://fonts.googleapis.com/css2?family=Kaisei+Decol&family=Yomogi&family=Zen+Kurenaido&display=swap');
    </style>
    ```
### 4. レイアウト
  - ヘッダー：左揃えタイトル＋右揃え日付/出典
  - 3カラム構成：左側33%、中央33%、右側33%
  - カード型コンポーネント：白背景、角丸12px、微細シャドウ
  - セクション間の適切な余白と階層構造
  - 適切にグラスモーフィズムを活用
  - 横幅は100%にして
## グラフィックレコーディング表現技法
- テキストと視覚要素のバランスを重視
- キーワードを囲み線や色で強調
- 簡易的なアイコンや図形で概念を視覚化
- 数値データは簡潔なグラフや図表で表現
- 接続線や矢印で情報間の関係性を明示
- 余白を効果的に活用して視認性を確保
## 全体的な指針
- 読み手が自然に視線を移動できる配置
- 情報の階層と関連性を視覚的に明確化
- 手書き風の要素で親しみやすさを演出
- 視覚的な記憶に残るデザイン
- フッターに出典情報を明記
"""

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
        thinking={"type": "adaptive"},
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


async def process_message_with_attachments(message, attachments, cleaned_text, save_to_file):
    """Process a message with multiple attachments"""
    content = []
    
    # Add the text content first if it exists
    if cleaned_text:
        content.append({"type": "text", "text": cleaned_text})

    # Validation loop
    for attachment in attachments:
        mime_type = get_mime_type(attachment.filename)
        
        if not mime_type:
            # Simple warning for now, listing all might be too long with dynamic lookup
            await message.channel.send(f"🗑️ Unsupported file type/extension: {attachment.filename}")
            return # Stop processing if any file is invalid

    # Processing loop
    for attachment in attachments:
        file_extension = os.path.splitext(attachment.filename.lower())[1]
        mime_type = get_mime_type(attachment.filename)
        
        file_data = await download_attachment(attachment)
        if file_data is None:
            await message.channel.send(f'Unable to download the file: {attachment.filename}')
            return

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


# グラフィックレコーディングのプロンプトを作成
def create_graphic_recording_prompt(user_prompt, with_file=False):
    """グラフィックレコーディング用のプロンプトを作成"""
    base_prompt = GRAPHIC_RECORDING_TEMPLATE

    if with_file:
        file_instruction = f"""
## 変換する内容
添付されたファイルを分析し、その内容を理解してください。以下のプロンプトに基づいて、ファイルの内容をグラフィックレコーディングとしてまとめてください:
{user_prompt}

出力形式：完全なHTMLコードで返してください。```html ... ```の形式で返してください。HTMLにはすべてのスタイルを含め、外部リソースへの依存がないようにしてください。
"""
        return base_prompt + file_instruction
    else:
        text_instruction = f"""
## 変換する文章/記事
以下のプロンプトに基づいて、グラフィックレコーディングを作成してください:
{user_prompt}
これまでの会話履歴も考慮に入れてください。

出力形式：完全なHTMLコードで返してください。```html ... ```の形式で返してください。HTMLにはすべてのスタイルを含め、外部リソースへの依存がないようにしてください。
"""
        return base_prompt + text_instruction


# グラフィックレコーディング処理関数
async def process_graphic_recording(message, prompt, attachment=None):
    """グラフィックレコーディングを生成する関数"""
    await message.add_reaction('📊')  # リアクションを追加してコマンド受付を示す
    
    async with message.channel.typing():
        if attachment:
            # 添付ファイルがある場合の処理
            file_extension = os.path.splitext(attachment.filename.lower())[1]
            mime_type = get_mime_type(attachment.filename)
            
            if not mime_type:
                await message.channel.send(f"🗑️ サポートされていないファイル形式です。")
                return
            
            file_data = await download_attachment(attachment)
            if file_data is None:
                await message.channel.send(f'ファイルのダウンロードに失敗しました: {attachment.filename}')
                return
            
            # ファイルタイプに応じた処理
            enhanced_prompt = create_graphic_recording_prompt(prompt, with_file=True)
            
            if mime_type.startswith('image/'):
                # 画像ファイルの処理
                resized_image_stream = resize_image(file_data, file_extension)
                encoded_data = base64.b64encode(resized_image_stream.getvalue()).decode("utf-8")
                content = [
                    {"type": "text", "text": enhanced_prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}}
                ]
            elif mime_type == 'application/pdf':
                # PDFファイルの処理
                encoded_data = base64.b64encode(file_data).decode("utf-8")
                content = [
                    {"type": "text", "text": enhanced_prompt},
                    {"type": "document", "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}}
                ]
            else:
                # テキストベースのファイル処理
                try:
                    text_data = file_data.decode('utf-8', errors='replace')
                    file_info = (
                        f"## File Information\n"
                        f"- Name: `{attachment.filename}`\n"
                        f"- Size: {len(text_data)} characters\n"
                        f"- Type: {mime_type}\n\n"
                    )
                    full_prompt = f"{enhanced_prompt}\n\n{file_info}{text_data}"
                    content = full_prompt
                except:
                    await message.channel.send(f"このファイル形式は処理できません。")
                    return
            
            # 共通の処理部分
            if isinstance(content, list):
                update_history(message.author.id, content, 'user')
            else:
                update_history(message.author.id, content, 'user')
                
            formatted_history = get_history(message.author.id)
            thinking_text, response_text = await generate_response(formatted_history)
            update_history(message.author.id, response_text, "assistant")
            
            # HTMLの抽出と処理
            await process_html_response(message, response_text)
        else:
            # テキストのみの処理
            enhanced_prompt = create_graphic_recording_prompt(prompt, with_file=False)
            update_history(message.author.id, enhanced_prompt, 'user')
            formatted_history = get_history(message.author.id)
            thinking_text, response_text = await generate_response(formatted_history)
            update_history(message.author.id, response_text, "assistant")
            
            # HTMLの抽出と処理
            await process_html_response(message, response_text)


async def process_html_response(message, response_text):
    """HTMLレスポンスを処理してDiscordに送信する関数"""
    try:
        # HTMLコードを抽出
        html_match = re.search(r'```html\s*([\s\S]*?)\s*```', response_text)
        if not html_match:
            # HTML形式でない場合は通常のテキストとして送信
            await message.channel.send("グラフィックレコーディングの生成に失敗しました。HTMLコードが見つかりません。")
            await send_long_message(message, response_text, MAX_DISCORD_LENGTH)
            return
            
        html_code = html_match.group(1)
        
        # HTMLをファイルとして保存
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"graphic_recording_{timestamp}.html"
        
        # HTMLファイルを作成して送信
        html_file = discord.File(io.StringIO(html_code), filename=filename)
        await message.channel.send(f"🎨 グラフィックレコーディングが完成しました！", file=html_file)
        
        # Embedとしても表示
        await send_graphic_recording_preview(message, html_code, response_text)
        
    except Exception as e:
        await message.channel.send(f"HTMLの処理中にエラーが発生しました: {e}")
        await send_long_message(message, response_text, MAX_DISCORD_LENGTH)


async def send_graphic_recording_preview(message, html_code, full_response):
    """グラフィックレコーディングのプレビューをEmbed形式で表示"""
    try:
        # タイトルを抽出
        title_match = re.search(r'<h1[^>]*>(.*?)<\/h1>', html_code, re.DOTALL)
        title = title_match.group(1) if title_match else "グラフィックレコーディング"
        title = re.sub(r'<[^>]+>', '', title)  # HTMLタグを削除
        
        # 説明を抽出（最初の段落またはdivの内容）
        desc_match = re.search(r'<p[^>]*>(.*?)<\/p>|<div[^>]*>(.*?)<\/div>', html_code, re.DOTALL)
        description = desc_match.group(1) if desc_match and desc_match.group(1) else desc_match.group(2) if desc_match else "内容のプレビュー"
        
        # HTML要素のタグを削除してプレーンテキスト化
        description = re.sub(r'<[^>]+>', '', description)
        # Discordのembedの説明は最大4096文字まで
        description = description[:2000] + "..." if len(description) > 2000 else description
        
        # Embedを作成
        embed = discord.Embed(
            title=title[:256],  # タイトルは256文字まで
            description=description,
            color=0xF25C05  # テンプレートの「ファッション-4」カラー
        )
        
        # キーポイントを抽出（リスト要素など）
        list_items = re.findall(r'<li[^>]*>(.*?)<\/li>', html_code, re.DOTALL)
        if list_items:
            # 制限内に収まるようにキーポイントを取得
            key_points = []
            points_text = ""
            for item in list_items:
                plain_text = re.sub(r'<[^>]+>', '', item).strip()
                if plain_text:
                    new_point = f"• {plain_text}\n"
                    # フィールド値の制限は1024文字
                    if len(points_text + new_point) > 1000:  # 余裕を持たせる
                        points_text += "..."
                        break
                    points_text += new_point
                    key_points.append(plain_text)
            
            if points_text:
                embed.add_field(
                    name="🔑 キーポイント",
                    value=points_text[:1024],  # 確実に制限内に収める
                    inline=False
                )
        
        # 見出しを抽出
        headings = re.findall(r'<h[2-4][^>]*>(.*?)<\/h[2-4]>', html_code, re.DOTALL)
        if headings:
            # 制限内に収まるように見出しを取得
            headings_text = ""
            processed_headings = []
            
            for h in headings:
                plain_heading = re.sub(r'<[^>]+>', '', h).strip()
                if plain_heading:
                    new_heading = f"📌 {plain_heading}\n"
                    # フィールド値の制限は1024文字
                    if len(headings_text + new_heading) > 1000:  # 余裕を持たせる
                        headings_text += "..."
                        break
                    headings_text += new_heading
                    processed_headings.append(plain_heading)
            
            if headings_text:
                embed.add_field(
                    name="📋 セクション",
                    value=headings_text[:1024],  # 確実に制限内に収める
                    inline=False
                )
        
        # フッター
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embed.set_footer(text=f"Graphic Recording | {timestamp}")
        
        await message.channel.send(embed=embed)
        
    except Exception as e:
        await message.channel.send(f"プレビューの作成中にエラーが発生しました: {str(e)}")
        # HTML全体をプレビューとして送信せず、エラーのみを表示


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


# コマンドとして!graを実装
@bot.command(name='gra')
async def gra_command(ctx, *, prompt=None):
    """Generate a graphic recording based on the prompt."""
    prompt = prompt or ""
    attachment = ctx.message.attachments[0] if ctx.message.attachments else None
    await process_graphic_recording(ctx.message, prompt, attachment)


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