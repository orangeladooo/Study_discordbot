import json          # Built-in Python module to read/write JSON files
import os            # Built-in module to check if files exist
from datetime import datetime  # To timestamp every message we save

# ─────────────────────────────────────────────────────────────────────────────
# WHERE MEMORY IS STORED
# ─────────────────────────────────────────────────────────────────────────────
# This file will be created automatically next to memory.py when the bot runs.
# It holds EVERYTHING the bot remembers about you — conversations + profile.
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")

# ─────────────────────────────────────────────────────────────────────────────
# HOW MUCH CONVERSATION HISTORY TO KEEP
# ─────────────────────────────────────────────────────────────────────────────
# We keep the last 20 messages. Any older ones are dropped to save tokens.
# 20 is a good balance — enough context without overloading the LLM.
MAX_HISTORY = 20

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
# If memory.json doesn't exist yet, we start with this blank slate.
# "history"  → list of past messages (human + AI turns)
# "profile"  → dict of facts we learn about the user over time
DEFAULT_MEMORY = {
    "history": [],
    "profile": {
        "learning_style": "",          # e.g. "prefers visual examples"
        "strong_topics": [],           # e.g. ["arrays", "binary search"]
        "weak_topics": [],             # e.g. ["dynamic programming"]
        "preferred_explanation": "",   # e.g. "explain with analogies"
        "notes": []                    # Any freeform facts about the user
    }
}


def load_memory() -> dict:
    """
    Reads memory.json from disk and returns it as a Python dict.
    If the file doesn't exist yet (first run), returns the DEFAULT_MEMORY above.
    """
    if not os.path.exists(MEMORY_FILE):
        # First time running — no memory file yet, return a clean slate
        return DEFAULT_MEMORY.copy()

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        # json.load() reads the file and converts JSON → Python dict
        return json.load(f)


def save_memory(memory: dict):
    """
    Writes the memory dict to memory.json on disk.
    indent=2 makes the JSON human-readable (nice formatting).
    ensure_ascii=False lets us save non-English characters properly.
    """
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_message(role: str, content: str):
    """
    Adds a single message to conversation history, then saves to disk.

    role    → either "human" (your messages) or "ai" (bot responses)
    content → the actual text of the message

    We also trim history to MAX_HISTORY so the file doesn't grow forever.
    """
    memory = load_memory()

    # Build the message entry with a timestamp so we know when it happened
    message = {
        "role": role,                          # "human" or "ai"
        "content": content,                    # The actual message text
        "timestamp": datetime.now().isoformat() # e.g. "2026-02-23T01:30:00"
    }

    memory["history"].append(message)

    # Keep only the most recent MAX_HISTORY messages
    # Slicing from -MAX_HISTORY to end gives us the last 20 items
    if len(memory["history"]) > MAX_HISTORY:
        memory["history"] = memory["history"][-MAX_HISTORY:]

    save_memory(memory)


def get_history() -> list:
    """
    Returns the conversation history as a list of dicts.
    Each dict has: {"role": "human"/"ai", "content": "...", "timestamp": "..."}
    This is passed to the agent to give it context about past conversations.
    """
    memory = load_memory()
    return memory["history"]


def get_profile() -> dict:
    """
    Returns the user's learning profile dict.
    Used by the agent to personalize its system prompt.
    """
    memory = load_memory()
    return memory["profile"]


def update_profile(key: str, value):
    """
    Updates a single field in the user's profile.

    Examples:
      update_profile("learning_style", "prefers short bullet points")
      update_profile("weak_topics", ["DP", "graphs"])
      update_profile("notes", ["Adwith is in 3rd year CS"])

    If the field holds a list and value is also a list, we MERGE them
    (no duplicates). If it's a string, we just overwrite.
    """
    memory = load_memory()

    if key in memory["profile"]:
        existing = memory["profile"][key]

        # If both old and new are lists, merge them and remove duplicates
        if isinstance(existing, list) and isinstance(value, list):
            combined = existing + value
            # dict.fromkeys() preserves order while removing duplicates
            memory["profile"][key] = list(dict.fromkeys(combined))
        else:
            # For strings (like learning_style), just overwrite
            memory["profile"][key] = value

    save_memory(memory)


def remember_note(note: str):
    """
    Adds a freeform fact to the user's notes list.
    Example: remember_note("prefers code examples over theory")
    This is called when the user runs /remember command.
    """
    memory = load_memory()
    # Only add if it's not already in the list (avoid duplicates)
    if note not in memory["profile"]["notes"]:
        memory["profile"]["notes"].append(note)
    save_memory(memory)


def clear_history():
    """
    Wipes the conversation history but KEEPS the user profile.
    Called when user runs /clear command.
    """
    memory = load_memory()
    memory["history"] = []   # Empty list = no messages
    save_memory(memory)


def format_profile_for_display() -> str:
    """
    Converts the profile dict into a nicely formatted string for Discord.
    Used by the /profile command to show what the bot knows about you.
    """
    profile = get_profile()
    lines = ["**📋 Your Learning Profile**\n"]

    if profile["learning_style"]:
        lines.append(f"🎯 **Learning Style:** {profile['learning_style']}")

    if profile["preferred_explanation"]:
        lines.append(f"💡 **Preferred Explanations:** {profile['preferred_explanation']}")

    if profile["strong_topics"]:
        lines.append(f"💪 **Strong Topics:** {', '.join(profile['strong_topics'])}")

    if profile["weak_topics"]:
        lines.append(f"📚 **Topics to Work On:** {', '.join(profile['weak_topics'])}")

    if profile["notes"]:
        lines.append("\n📝 **Personal Notes:**")
        for note in profile["notes"]:
            lines.append(f"  • {note}")

    if len(lines) == 1:
        # Only the header — profile is empty
        return "Your profile is empty. Chat with me a bit and I'll learn about you! You can also use `/remember` to tell me things directly."

    return "\n".join(lines)
