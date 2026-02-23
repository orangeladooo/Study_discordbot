# 🤖 StudyBot — Personal AI Tutor Discord Bot

A personalized Discord bot powered by **Groq** that acts as your private AI tutor. Built for DSA practice, exam prep, and general learning.

---

## ✨ Features

- 🧠 **Smart model routing** — auto-picks coding or general model based on your query
- 💾 **Persistent memory** — remembers your profile and conversation history across sessions
- 🌐 **Web search** — searches the web live for current info when needed
- 📄 **Multimodal** — reads PDFs, describes images, and extracts video frames via `$see`
- 🎓 **Tutor persona** — short answers, real-life analogies, always checks your understanding

---

## 🚀 Setup

### 1. Clone & install
```bash
git clone https://github.com/yourusername/studybot.git
cd studybot
pip install -r bot/requirements.txt
```

### 2. Create `bot/.env`
```env
DISCORD_TOKEN=your_discord_bot_token_here
GROQ_API_KEY=your_groq_api_key_here
CODER_MODEL=qwen/qwen3-32b
GENERAL_MODEL=llama-3.3-70b-versatile
VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```
- Discord token → [discord.com/developers/applications](https://discord.com/developers/applications)
- Groq API key (free) → [console.groq.com](https://console.groq.com)

### 3. Enable Message Content Intent
Developer Portal → your app → **Bot** → enable **Message Content Intent**

### 4. Run
```bash
cd bot && python bot.py
```

---

## 💬 Commands

| Command | Description |
|---|---|
| `$ask [question]` | Ask anything |
| `$ask --code [q]` | Force coding model |
| `$ask --think [q]` | Force general model |
| `$search [query]` | Search the web and summarize |
| `$see [question]` | Attach a PDF / image / video and ask about it |
| `$remember [fact]` | Tell the bot something about yourself |
| `$profile` | View your saved profile |
| `$clear` | Clear conversation history |
| `$help` | Show all commands |

