# Discord bot with Claude (Adaptive Thinking) on Google Cloud Vertex AI

Claude Bot is a sophisticated Discord bot that leverages Google Cloud's Vertex AI and the latest Claude models (e.g. Claude Opus 4.8 / Sonnet 5) to interact with users through text, image, and PDF responses. The model is configurable via the `MODEL` environment variable, so the bot can run on any Claude model in Vertex AI that supports adaptive thinking (e.g. Opus 4.8 / Sonnet 5) — the bot always requests adaptive thinking, so models without it will error. It's designed to enhance Discord servers by providing intelligent and contextual interactions.

## Features

- **Advanced Text Processing**: Utilizes the latest Claude models (Opus 4.8 / Sonnet 5, selectable via the `MODEL` variable) for generating meaningful text responses.
- **Adaptive Thinking**: Uses Claude's Adaptive Thinking mode for optimized reasoning, letting the model decide when and how much to think.
- **Thinking Transparency**: With the `!save` command, the bot requests `display: "summarized"` and, when Claude produces a thinking summary, saves it as a separate file.
- **Graceful Error & Refusal Handling**: On an API error or an API-level refusal (`stop_reason == "refusal"`), the bot notifies the user instead of sending an empty message, and never persists a failed turn to the conversation history.
- **Image Recognition**: Can analyze and respond to images sent by users.
- **PDF Processing**: Can process and understand PDF documents.
- **Supports Multiple Programming Languages**: Can process various programming language source code files and provide relevant responses.
- **Environmentally Friendly**: Uses environment variables for secure configuration.
- **Customizable and Scalable**: Easy to adjust settings and scale for different use cases.
- **Conversation History**: Remembers previous interactions (configurable length) to provide context-aware responses.
- **Save Responses**: Users can save the bot's responses as text files directly in Discord using the `!save` command.
- **Multi-File Support**: Can process multiple attachments (images, PDFs, text files) in a single message.


## Supported File Extensions

The bot uses Python's standard `mimetypes` module to automatically detect file types. It generally supports:

-   **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`
-   **Documents**: `.pdf`
-   **Text/Code**: Most text-based files are automatically supported. Explicit support is added for:
    -   **Web**: `.html`, `.css`, `.js`, `.ts`, `.php`
    -   **Data/Config**: `.json`, `.csv`, `.xml`, `.yaml`, `.yml`, `.ini`, `.conf`, `.env`, `.sql`
    -   **Code**: `.py`, `.c`, `.h`, `.cpp`, `.hpp`, `.rs`, `.cs`, `.rb`, `.pl`, `.pm`, `.swift`, `.r`, `.go`, `.kt`, `.java`, `.lua`
    -   **Scripts**: `.sh`, `.bat`
    -   **Docs**: `.md`, `.txt`

## Prerequisites

-   Python 3.9 or newer (required by the Anthropic SDK; tested on Python 3.12).
-   A Google Cloud account and project setup for Vertex AI.
-   A Discord bot token.

## Installation

1.  **Clone the Repository**

    Start by cloning the repository to your local machine:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Required Packages**

    Install the Python dependencies listed in `requirements.txt` (this includes
    `anthropic[vertex]` for Vertex AI integration, `discord.py`, `python-dotenv`,
    `Pillow`, and `aiohttp`):

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**

    Create a `.env` file in the root directory of the project. You can copy the provided example file:

    ```bash
    cp .env.example .env
    ```

    Then, open `.env` and populate it with the necessary environment variables:

    ```
    DISCORD_BOT_TOKEN=your_discord_bot_token_here
    GCP_REGION=global
    GCP_PROJECT_ID=your_gcp_project_id
    MAX_HISTORY=10
    MODEL=claude-opus-4-8
    SYSTEM_PROMPT=
    WEB_SEARCH_MAX_USES=0
    ```

    Replace the placeholders with your actual information.  `MAX_HISTORY` refers to the number of *pairs* of user/assistant messages to store.  `MODEL` is the bare Vertex AI model ID to use — e.g., `claude-opus-4-8` (most capable) or `claude-sonnet-5` (balanced speed/cost). The bot always requests adaptive thinking, so the model must support it (Opus 4.6+/4.7/4.8, Sonnet 5, Sonnet 4.6); models without adaptive thinking will return a 400 error. `SYSTEM_PROMPT` is optional — set a string to customize the bot's persona/behavior, or leave empty to send no system prompt. `WEB_SEARCH_MAX_USES` controls Claude's built-in web search: set to 0 to disable (default), or a positive integer to allow Claude to perform up to that many web searches per response (billed at $10 per 1,000 searches by Anthropic). Source URLs are appended to responses as a Markdown list.



4.  **Run the Bot**

    With the setup complete, you can start the bot using:

    ```bash
    python Stream_ClaudeDiscordBot.py
    ```

    *Note: `ClaudeDiscordBot.py` has been deprecated and renamed to `ClaudeDiscordBot_deprecated.py`. Please use `Stream_ClaudeDiscordBot.py` for the latest features.*

## Usage

Once the bot is running, it will listen for messages in all the servers it has been added to. Users can interact with the bot by:

-   Mentioning the bot in a message (`@BotName`).
-   Sending a direct message (DM) to the bot.
-   Uploading supported file types (images, PDFs, text-based files).
-   **Sending multiple files**: You can attach multiple files (images, PDFs, text) in a single message. The bot will process all of them.
-   Using the `!save` command at the beginning of a message to save the response as a file. When Claude produces a thinking summary, it is saved as a separate `claude_thinking_*.md` file alongside the `claude_response_*.md` file.


The bot can process text, images, and PDFs, providing responses generated by Claude. You can reset the conversation history by sending `RESET` (case-insensitive) to the bot. If the API returns an API-level refusal (`stop_reason == "refusal"`) or an API error occurs, the bot replies with a short notice instead of failing silently.


## Customization

Modify the `.env` file to change environmental settings such as the Discord bot token, maximum message history, and Google Cloud configurations. The code itself can be customized to alter the bot's behavior, command prefixes, and more.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
