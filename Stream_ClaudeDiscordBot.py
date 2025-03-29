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
intents = discord.Intents.default()
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


# グラフィックレコーディングのプロンプトを作成
def create_graphic_recording_prompt(user_prompt, with_file=False):
    """グラフィックレコーディング用のプロンプトを作成"""
    base_prompt = GRAPHIC_RECORDING_TEMPLATE

    if with_file:
        file_instruction = f"""
## 変換する文章/記事
添付されたPDFファイルを分析し、その内容を理解してください。以下のプロンプトに基づいて、PDFの内容をグラフィックレコーディングとしてまとめてください:
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
            mime_type = ext_to_mime.get(file_extension)
            
            if not mime_type:
                await message.channel.send(f"🗑️ サポートされていないファイル形式です。")
                return
            
            file_data = await download_attachment(attachment)
            if file_data is None:
                await message.channel.send(f'ファイルのダウンロードに失敗しました: {attachment.filename}')
                return
            
            # PDFファイルの処理
            if file_extension == '.pdf':
                encoded_data = base64.b64encode(file_data).decode("utf-8")
                enhanced_prompt = create_graphic_recording_prompt(prompt, with_file=True)
                
                content = [
                    {"type": "text", "text": enhanced_prompt},
                    {"type": "document", "source": {"type": "base64", "media_type": mime_type, "data": encoded_data}}
                ]
                
                update_history(message.author.id, content, 'user')
                formatted_history = get_history(message.author.id)
                thinking_text, response_text = await generate_response(formatted_history)
                update_history(message.author.id, response_text, "assistant")
                
                # HTMLの抽出と処理
                await process_html_response(message, response_text)
            else:
                await message.channel.send(f"グラフィックレコーディングにはPDFファイルのみ対応しています。")
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
            for attachment in ctx.message.attachments:
                await process_attachment(ctx.message, attachment, cleaned_text, True)
                break  # Process only the first attachment
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
                for attachment in message.attachments:
                    await process_attachment(message, attachment, cleaned_text, False)
                    break  # Process only the first attachment
            else:
                await process_text(message, cleaned_text, False)


bot.run(DISCORD_BOT_TOKEN)