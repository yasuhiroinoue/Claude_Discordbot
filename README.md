
# Discord bot with Claude-3 using Google Cloud VertexAI

Claude Bot is a sophisticated Discord bot that leverages Google Cloud's VertexAI and the Claude-3 language model to interact with users through text and image responses. It's designed to enhance Discord servers by providing intelligent and contextual interactions.

## Features

- **Advanced Text Processing**: Utilizes Claude-3, a powerful language model, for generating meaningful text responses.
- **Image Recognition**: Can analyze and respond to images sent by users.
- **Supports Multiple Programming Languages**: Can process various programming language source code files and provide relevant responses.
- **Environmentally Friendly**: Uses environment variables for secure configuration.
- **Customizable and Scalable**: Easy to adjust settings and scale for different use cases.

## Supported File Extensions

The bot now supports a wide range of text-based file formats, including but not limited to:
- Text files: `.txt`
- Markdown files: `.md`
- CSV files: `.csv`
- JSON files: `.json`
- XML files: `.xml`
- HTML files: `.html`
- Configuration files: `.ini`, `.log`, `.yaml`, `.yml`
- Source code files:
  - C: `.c`, `.h`
  - C++: `.cpp`, `.hpp`
  - Python: `.py`
  - Rust: `.rs`
  - JavaScript: `.js`
  - C#: `.cs`
  - PHP: `.php`
  - Ruby: `.rb`
  - Perl: `.pl`, `.pm`
  - Swift: `.swift`
  - R: `.R`, `.r`
  - Go: `.go`

## Prerequisites

- Python 3.8 or newer.
- A Google Cloud account and project setup for VertexAI.
- A Discord bot token.

## Installation

1. **Clone the Repository**

   Start by cloning the repository to your local machine:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Required Packages**

   Install the Python dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Additionally, install the `anthropic[vertex]` package for VertexAI integration:

   ```bash
   python -m pip install -U 'anthropic[vertex]'
   ```

3. **Configure Environment Variables**

   Create a `.env` file in the root directory of the project and populate it with the necessary environment variables:

   ```
   DISCORD_BOT_TOKEN=<Your Discord Bot Token>
   MAX_HISTORY=<Maximum Number of Messages to Remember>
   GCP_REGION=<Google Cloud Platform Region>
   GCP_PROJECT_ID=<Your GCP Project ID>
   ```

   Replace the placeholders with your actual information.

4. **Run the Bot**

   With the setup complete, you can start the bot using:

   ```bash
   python ClaudeDiscordBot.py
   ```

## Usage

Once the bot is running, it will listen for messages in all the servers it has been added to. Users can interact with the bot by mentioning it in a message or sending a direct message. The bot can process both text and images, providing responses generated by Claude-3.

## Customization

Modify the `.env` file to change environmental settings such as the Discord bot token, maximum message history, and Google Cloud configurations. The code itself can be customized to alter the bot's behavior, command prefixes, and more.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
