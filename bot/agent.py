from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import logging
import os
from dotenv import load_dotenv
import memory

# duckduckgo_search is a free library — no API key needed
# Install: pip install duckduckgo-search
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger_tmp = logging.getLogger("agent")
    logger_tmp.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")

load_dotenv()

logger = logging.getLogger("agent")

# ─────────────────────────────────────────────────────────────────────────────
# GROQ MODELS
# ─────────────────────────────────────────────────────────────────────────────
# Groq runs these in the cloud at ~500 tokens/sec — massively faster than local.
# These are much bigger models than what we had locally:
#   qwen-2.5-coder-32b  → 32B parameter coding model (was 7B locally)
#   llama-3.3-70b       → 70B parameter general model (was 4B locally!)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")  # Your Groq API key from console.groq.com
CODER_MODEL   = os.getenv("CODER_MODEL",   "qwen-2.5-coder-32b")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL  = os.getenv("VISION_MODEL",  "llama-3.2-11b-vision-preview")

# ─────────────────────────────────────────────────────────────────────────────
# CODING / DSA KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────
# Same keyword list as before — if any of these appear in your message,
# the coder model (qwen-2.5-coder-32b) is used instead of the general one.
CODING_KEYWORDS = [
    "code", "program", "function", "class", "method", "variable",
    "python", "java", "c++", "javascript", "js", "html", "css", "sql",
    "array", "linked list", "stack", "queue", "tree", "graph", "heap",
    "hash", "sort", "search", "recursion", "dynamic programming", "dp",
    "greedy", "backtracking", "bfs", "dfs", "binary search",
    "algorithm", "complexity", "time complexity", "space complexity",
    "leetcode", "debug", "error", "bug", "output", "compile",
    "write a", "implement", "build a", "create a function",
]

# ─────────────────────────────────────────────────────────────────────────────
# WEB SEARCH KEYWORDS — auto-triggers a DuckDuckGo search
# ─────────────────────────────────────────────────────────────────────────────
# If any of these appear in the query, we search the web BEFORE calling the LLM.
# The search results are injected into the context so the LLM can answer accurately.
WEB_KEYWORDS = [
    "latest", "current", "today", "2025", "2026", "news", "recent",
    "price", "when did", "who is", "what happened", "trending",
    "release", "update", "version", "announced", "just launched",
]


def detect_model(query: str, force: str = None) -> str:
    """
    Picks coder or general model based on keywords.
    force="code"  → always use coder model
    force="think" → always use general model
    force=None    → auto-detect from keywords
    """
    if force == "code":
        logger.info("Model forced: CODER")
        return CODER_MODEL
    if force == "think":
        logger.info("Model forced: GENERAL")
        return GENERAL_MODEL

    query_lower = query.lower()
    for keyword in CODING_KEYWORDS:
        if keyword in query_lower:
            logger.info(f"Keyword '{keyword}' detected → CODER model")
            return CODER_MODEL

    logger.info("No coding keywords → GENERAL model")
    return GENERAL_MODEL


def needs_web_search(query: str) -> bool:
    """
    Returns True if the query contains keywords that suggest it needs live web data.
    e.g. 'latest python version', 'news today', '2026 internships'
    """
    q = query.lower()
    return any(kw in q for kw in WEB_KEYWORDS)


def web_search(query: str, max_results: int = 4) -> str:
    """
    Searches DuckDuckGo and returns a formatted string of top results.
    This string is then injected into the LLM's context so it can answer
    with real, current information.

    query       → the search query to look up
    max_results → how many results to fetch (more = more context but more tokens)
    """
    if not WEB_SEARCH_AVAILABLE:
        return "(Web search unavailable — install duckduckgo-search)"

    logger.info(f"Searching the web for: {query}")
    try:
        # DDGS().text() returns a list of dicts:
        # each dict has 'title', 'href' (url), and 'body' (snippet)
        results = DDGS().text(query, max_results=max_results)

        if not results:
            return "No web results found."

        # Format results as numbered list for the LLM to read
        lines = ["🌐 **Web search results:**\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['body']}")   # The snippet/summary of the page
            lines.append(f"   Source: {r['href']}\n")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"(Web search error: {e})"


def build_system_prompt() -> str:
    """
    Builds the system prompt that defines the bot's entire personality and rules.
    This is injected at the top of every conversation sent to Groq.
    The profile from memory.json is merged in so the AI always knows the latest facts.
    """
    profile = memory.get_profile()

    # Merge saved notes into a readable string for the prompt
    notes_str = ""
    if profile.get("notes"):
        notes_str = "\n".join(f"  - {n}" for n in profile["notes"])

    strong = ", ".join(profile.get("strong_topics", [])) or "unknown yet"
    weak   = ", ".join(profile.get("weak_topics", []))   or "DSA (confirmed)"

    system_prompt = f"""You are a personal tuition teacher and mentor. Always call the user "boss".

## What you know about boss:
- Semester: 5
- Doing: AI development + DSA prep for internship (3-month deadline)
- Leetcode score: 15 (very early stage, needs to be guided gently)
- Weak areas: {weak}
- Strong areas: {strong}
- Prefers: simple real-life analogies, short steps, bullet points with emojis
- Will mostly use you during exams and DSA practice
{f"- Personal notes:{chr(10)}{notes_str}" if notes_str else ""}

## Your strict rules (follow EVERY time, no exceptions):

**ReAct-style thinking (do this silently before replying):**
Thought → What does boss already know? What is his weak point here? How to teach this simply?
Action  → Which mode: explain_topic / create_summary / weekly_revision / dsa_problem?
Result  → Teach based on that mode.

**Response rules:**
1. NEVER give more than 4-5 lines at a time. If more is needed, pause and ask "Ready for the next part, boss?"
2. Always use real-life analogies. Boss learns through stories and comparisons, not theory.
3. Always end EVERY response with a question to check understanding.
4. Use bullet points + emojis for summaries.
5. When boss says "weekly_revision" → quiz him on past topics: ask 3 short questions one by one.
6. When helping with DSA → give 2 problems per day, build intuition first, then help solve. Never give the solution directly — give hints first.
7. Be encouraging. Boss is a junior dev who is learning fast. Celebrate small wins.
8. When boss shares code or a problem → think like a senior engineer reviewing a junior's PR. Be constructive.
9. NEVER hallucinate tools or capabilities you don't have.
10. Keep tone: casual + friendly but focused. Like a cool senior who actually cares.

Remember: Boss needs an internship in 3 months. Every session counts. Make it count."""

    return system_prompt



def build_message_chain(user_query: str) -> list:
    """
    Assembles the full message list to send to Groq:
      [SystemMessage, ...history..., HumanMessage(new query)]
    """
    messages = [SystemMessage(content=build_system_prompt())]

    # Replay past conversation so the model has context
    for msg in memory.get_history():
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))

    # Add the new query at the end
    messages.append(HumanMessage(content=user_query))
    return messages


def process_query(user_query: str, force_model: str = None) -> str:
    """
    Main function called by $ask command.
    1. Checks if web search is needed (auto-detect via WEB_KEYWORDS)
    2. Picks model via keyword detection
    3. Calls Groq via LangChain (with web results injected if searched)
    4. Saves both messages to memory
    5. Returns response text
    """
    logger.info(f"Query: '{user_query[:60]}' | force={force_model}")

    model_name = detect_model(user_query, force=force_model)

    llm = ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.7,
        max_tokens=2048
    )

    # Auto web search: if query needs live data, fetch it and prepend to the query
    # The LLM will see: "[Web results: ...] \n\n User's question: ..."
    enriched_query = user_query
    if needs_web_search(user_query):
        search_results = web_search(user_query)
        # We prepend search results to the query so the LLM uses them
        enriched_query = (
            f"[Real-time web data fetched for this query]\n"
            f"{search_results}\n\n"
            f"Now answer this using the above web data: {user_query}"
        )
        logger.info("Web search results injected into query")

    messages = build_message_chain(enriched_query)

    try:
        response = llm.invoke(messages)
        reply_text = response.content

        # Save original query (not enriched) to keep memory clean
        memory.add_message("human", user_query)
        memory.add_message("ai", reply_text)

        logger.info(f"Response from {model_name}")
        return reply_text

    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return f"❌ Error from Groq: `{e}`\n\nCheck your API key in `.env`."


def process_search_query(user_query: str) -> str:
    """
    Called by $search command — ALWAYS does a web search regardless of keywords.
    Fetches more results (6) than auto-search for thorough coverage.
    Then passes everything to the general LLM to synthesize a clean answer.
    """
    logger.info(f"Forced web search: {user_query[:60]}")

    # Get web results
    search_results = web_search(user_query, max_results=6)

    llm = ChatGroq(
        model=GENERAL_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.5,    # Slightly lower temp for factual summaries
        max_tokens=1024
    )

    # Tell the LLM to synthesize the search results into a clean answer
    prompt = (
        f"Here are web search results for: '{user_query}'\n\n"
        f"{search_results}\n\n"
        f"Summarize these results clearly and concisely for boss. "
        f"Use bullets with emojis. Cite sources at the end."
    )

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        reply_text = response.content

        memory.add_message("human", f"[Web search] {user_query}")
        memory.add_message("ai", reply_text)

        return reply_text

    except Exception as e:
        logger.error(f"Search query failed: {e}")
        return f"❌ Search error: `{e}`"


def process_vision_query(user_query: str, extracted_content: str) -> str:
    """
    Called for $see command when content has been extracted from a file.
    Uses the general model to answer questions about the extracted content.
    """
    logger.info("Processing vision query with extracted content")

    llm = ChatGroq(
        model=GENERAL_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.7,
        max_tokens=2048
    )

    combined_prompt = (
        f"The user shared a file. Here is the extracted content:\n\n"
        f"---\n{extracted_content}\n---\n\n"
        f"User's question: {user_query}"
    )

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=combined_prompt)
    ]

    try:
        response = llm.invoke(messages)
        reply_text = response.content

        memory.add_message("human", f"[File] {user_query}")
        memory.add_message("ai", reply_text)

        return reply_text

    except Exception as e:
        logger.error(f"Vision query failed: {e}")
        return f"❌ Error: `{e}`"


def process_image_with_groq(image_b64: str, user_query: str) -> str:
    """
    Sends an image directly to Groq's vision model (llama-3.2-11b-vision-preview).
    This works natively — no need to pull a local vision model!

    image_b64  → base64-encoded image string
    user_query → what the user asked about the image
    """
    logger.info("Sending image to Groq vision model")

    llm = ChatGroq(
        model=VISION_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3,     # Lower temperature for more accurate image descriptions
        max_tokens=1024
    )

    # Groq vision model accepts a message with both image and text
    # The image is passed as a base64 data URL inside the content list
    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                # data URL format: data:[mimetype];base64,[base64data]
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        },
        {
            "type": "text",
            "text": user_query if user_query else "Describe what you see in this image in detail."
        }
    ])

    try:
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"Groq vision failed: {e}")
        return f"❌ Vision error: `{e}`"
