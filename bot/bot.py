import discord                           # The main Discord library
from discord.ext import commands         # Adds prefix command support ($ask, $see etc.)
import os                                # To read environment variables
from dotenv import load_dotenv           # Reads our .env file automatically
import logging                           # Terminal output / debugging
import asyncio                           # Python's async library

import agent   # Our agent.py (LLM routing + response generation)
import memory  # Our memory.py (history + profile storage)
import vision  # Our vision.py (PDF/image/video processing)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("bot")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN not found in .env file!")

# ─────────────────────────────────────────────────────────────────────────────
# BOT SETUP
# ─────────────────────────────────────────────────────────────────────────────
# command_prefix="$" means all commands start with $
# e.g. $ask, $see, $remember, $profile, $clear
#
# Intents: we need message_content=True so the bot can read what you type
# dm_messages=True is included in default intents already
intents = discord.Intents.default()
intents.message_content = True   # Required to read message text (Discord API v10+)

bot = commands.Bot(command_prefix="$", intents=intents)

# Remove the default help command so we can make our own cleaner one
bot.remove_command("help")


# ─────────────────────────────────────────────────────────────────────────────
# BOT READY EVENT
# ─────────────────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    """
    Runs when bot successfully connects to Discord.
    No slash command sync needed — prefix commands work instantly.
    """
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")

    # Set the bot's status (the text under its name in Discord)
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="$ask | $see | $help"
        )
    )
    print(f"\n✅ Bot is online! Logged in as {bot.user.name}")
    print("Prefix: $  |  Commands: $ask  $see  $remember  $profile  $clear  $help\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SEND LONG MESSAGES
# ─────────────────────────────────────────────────────────────────────────────
async def send_long(ctx, text: str):
    """
    Discord has a 2000 character limit per message.
    This splits a long response into chunks and sends each one.

    ctx  → the command context (contains the channel to reply to)
    text → the full text to send
    """
    LIMIT = 1990  # Slightly under 2000 to be safe

    chunks = [text[i:i+LIMIT] for i in range(0, len(text), LIMIT)]
    for chunk in chunks:
        await ctx.send(chunk)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: $ask
# ─────────────────────────────────────────────────────────────────────────────
@bot.command(name="ask")
async def ask_command(ctx, *, question: str = None):
    """
    Chat with the AI. Auto-routes to coder or general model.

    Usage:
      $ask what is a binary tree
      $ask write bubble sort in python
      $ask --code explain time complexity     ← force coder model
      $ask --think what is photosynthesis     ← force general model

    The * in `*, question` means "capture everything after $ask as one string"
    So $ask what is a tree  →  question = "what is a tree"
    """
    # If user typed $ask with nothing after it
    if not question:
        await ctx.send("❓ Usage: `$ask [your question]`\nExample: `$ask what is recursion`")
        return

    # Parse force-model flags
    force_model = None
    if question.startswith("--code "):
        force_model = "code"
        question = question[7:].strip()   # Remove "--code " prefix
    elif question.startswith("--think "):
        force_model = "think"
        question = question[8:].strip()   # Remove "--think " prefix

    # Show typing indicator while the LLM is generating
    # This shows "StudyBot is typing..." in Discord so you know it's working
    async with ctx.typing():
        loop = asyncio.get_event_loop()

        # run_in_executor runs the sync agent function in a background thread
        # so it doesn't freeze the bot while Ollama is thinking
        response = await loop.run_in_executor(
            None,               # Default thread pool
            agent.process_query,
            question,
            force_model
        )

    await send_long(ctx, response)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: $see
# ─────────────────────────────────────────────────────────────────────────────
@bot.command(name="see")
async def see_command(ctx, *, question: str = None):
    """
    Share a PDF, image, or video and ask about it.
    You MUST attach the file to the same message.

    Usage:
      $see summarize this          ← attach a PDF
      $see what is in this image   ← attach a photo
      $see what topics are covered ← attach a video
    """
    if not question:
        await ctx.send("❓ Usage: `$see [your question]` and attach a file.")
        return

    # ctx.message.attachments is a list of files the user attached
    if not ctx.message.attachments:
        await ctx.send("📎 Please attach a file (PDF, image, or video) along with your question.")
        return

    # We process only the first attachment if multiple are sent
    attachment = ctx.message.attachments[0]
    logger.info(f"$see: file={attachment.filename}, question={question[:50]}")

    async with ctx.typing():
        # Download the file as bytes (stays in memory, no disk write needed)
        file_bytes = await attachment.read()

        # Process in vision.py — extracts text or describes the visual
        extracted_content = await vision.process_attachment(
            file_bytes=file_bytes,
            filename=attachment.filename,
            user_question=question
        )

        # If it's an error message from vision.py, send directly
        if extracted_content.startswith("❌") or extracted_content.startswith("⚠️"):
            await ctx.send(extracted_content)
            return

        # Pass extracted content + question to the LLM for a proper answer
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            agent.process_vision_query,
            question,
            extracted_content
        )

    header = f"📎 **File:** `{attachment.filename}`\n\n"
    await send_long(ctx, header + response)

@bot.command(name="search")
async def search_command(ctx, *, query: str = None):
    """
    Force a web search on any topic and get a summarized answer.
    Always searches — even without keywords like 'latest' or 'today'.

    Usage:
      $search best DSA resources 2025
      $search latest python features
      $search internship tips for students
    """
    if not query:
        await ctx.send("❓ Usage: `$search [what to look up]`\nExample: `$search best resources to learn DSA`")
        return

    async with ctx.typing():
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            agent.process_search_query,
            query
        )

    await send_long(ctx, f"🔍 **Searched:** `{query}`\n\n{response}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: $remember
# ─────────────────────────────────────────────────────────────────────────────
@bot.command(name="remember")
async def remember_command(ctx, *, fact: str = None):
    """
    Tell the bot something about yourself directly.

    Usage:
      $remember I prefer short bullet point explanations
      $remember I am preparing for GATE 2027
      $remember I struggle with dynamic programming
    """
    if not fact:
        await ctx.send("❓ Usage: `$remember [a fact about you]`")
        return

    memory.remember_note(fact)
    logger.info(f"$remember: {fact[:60]}")

    await ctx.send(
        f"✅ Got it! I'll remember:\n**\"{fact}\"**\n"
        f"This will shape how I explain things to you going forward."
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: $profile
# ─────────────────────────────────────────────────────────────────────────────
@bot.command(name="profile")
async def profile_command(ctx):
    """
    Shows everything the bot has learned about you.
    Usage: $profile
    """
    profile_text = memory.format_profile_for_display()
    await ctx.send(profile_text)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: $clear
# ─────────────────────────────────────────────────────────────────────────────
@bot.command(name="clear")
async def clear_command(ctx):
    """
    Clears conversation history but keeps your profile.
    Usage: $clear
    """
    memory.clear_history()
    await ctx.send(
        "🗑️ Conversation history cleared!\n"
        "Your profile is still saved. Use `$profile` to see it."
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND: $help
# ─────────────────────────────────────────────────────────────────────────────
@bot.command(name="help")
async def help_command(ctx):
    """
    Shows all available commands.
    Usage: $help
    """
    help_text = """**📚 StudyBot Commands**

`$ask [question]` — Ask anything. Auto-picks coder or general model.
`$ask --code [q]` — Force the coding model (qwen2.5-coder)
`$ask --think [q]` — Force the general model (gemma3)

`$see [question]` — Attach a PDF/image/video + ask about it

`$remember [fact]` — Tell me something about yourself
`$profile` — See what I know about you
`$clear` — Clear chat history (keeps your profile)
`$help` — Show this message

**Examples:**
`$ask explain binary search`
`$ask write a merge sort in python`
`$see summarize this` *(+ attach PDF)*
`$remember I prefer examples over theory`"""

    await ctx.send(help_text)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLER
# ─────────────────────────────────────────────────────────────────────────────
@bot.event
async def on_command_error(ctx, error):
    """
    Catches any errors from commands and sends a friendly message
    instead of silently crashing.

    MissingRequiredArgument → user forgot to type their question
    CommandNotFound         → user typed $something that doesn't exist
    """
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❓ Missing input. Try `$help` to see how to use commands.")
    elif isinstance(error, commands.CommandNotFound):
        await ctx.send(f"❓ Unknown command. Type `$help` to see all commands.")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"❌ Something went wrong: `{error}`")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting bot...")
    bot.run(TOKEN)
