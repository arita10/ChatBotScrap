"""
chatbot_backend.py — Price Intelligence Chatbot Backend (Flask + OpenAI).

Architecture:
  - Flask REST API with a single POST /chat endpoint
  - OpenAI GPT-4o as the reasoning engine (function-calling mode)
  - chatbot_tools.py provides all Supabase data-access functions
  - The LLM picks the right tool, gets the data, then writes a friendly reply

Endpoints:
  POST /chat          { "message": "...", "session_id": "..." }  → { "reply": "..." }
  DELETE /chat        { "session_id": "..." }                    → clear history
  GET  /health        →  { "status": "ok" }
  GET  /deals         →  top 10 best deals right now (no LLM, raw data)

Run locally:
  python chatbot_backend.py
  # or with gunicorn:
  gunicorn chatbot_backend:app --bind 0.0.0.0:5000
"""

import json
import logging
import os
from collections import defaultdict
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from supabase import create_client, Client

import chatbot_tools as tools

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("price_chatbot")

app = Flask(__name__)
CORS(app)

# ── Clients ────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

# ── In-memory conversation history  (session_id → list of messages) ────────
_history: dict[str, list[dict]] = defaultdict(list)

MAX_HISTORY = 6         # keep last 3 turns (6 messages) — enough context, fewer tokens
MODEL = "gpt-4o-mini"   # ~15x cheaper than gpt-4o, accurate enough for price queries


# ─────────────────────────────────────────────────────────────────────────────
# Friendly Persona System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are 'Pro-Price Assistant' 🛒, a smart and friendly AI shopping assistant for Turkish grocery markets (BİM, A101, Şok, Migros, CarrefourSA).

IDENTITY: When asked "who are you?" or "what can you do?", introduce yourself warmly like this:
"Hi there! 👋 I'm Pro-Price Assistant, your personal AI shopping helper! 🛒
I help you save money on groceries by:
• 💰 Finding the cheapest prices across BİM, A101, Şok, Migros & CarrefourSA
• 📈 Tracking price trends so you know the best time to buy
• 🔥 Showing today's hottest deals and discounts
• 🍳 Suggesting ingredients for recipes with the best prices
Just ask me about any grocery product and I'll find the best deal for you! 😊"
(Adapt the language to match what the user wrote — Turkish or English.)

CRITICAL: Always reply in the EXACT same language the user wrote in. English→English, Turkish→Turkish. Never switch.

CRITICAL: Database has Turkish product names. Always translate keywords to Turkish before calling tools.
Translations: milk→süt, bread→ekmek, egg→yumurta, butter→tereyağı, cheese→peynir, oil→yağ, rice→pirinç, chicken→tavuk, water→su, yogurt→yoğurt

DOMAIN: You ONLY answer questions about:
- Grocery prices and where to buy
- Price trends and buy/wait advice
- Recipe ingredients and their prices (use suggest_for_recipe tool)
- Best deals today
If the user asks anything else (weather, news, jokes, general knowledge, etc.), politely decline and redirect: "I'm best at helping with grocery prices and shopping! 🛒 What product can I find for you today?"

PERSONALITY: Be warm, encouraging, and helpful. Use friendly language and relevant emojis (🛒💰🔥🎉👍⏳). Celebrate good deals enthusiastically!

Rules:
1. Always call a tool first. Never guess prices.
2. For recipe/ingredient questions → call suggest_for_recipe, then list each ingredient with cheapest price and market.
3. Interpret prices: is_at_lowest→"lowest price! 🎉 Buy now!"; drop>=10%→"Big deal! 🔥"; drop>=5%→"Good discount 👍"; above avg→"Consider waiting ⏳"
4. State the cheapest market and price.
5. Keep answers short and friendly.
6. No data found → apologize warmly, mention prices refresh daily at 07:00."""


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions (OpenAI function-calling schema)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_prices",
            "description": (
                "Get today's prices for a product across all markets, sorted cheapest first. "
                "Best for questions like 'Where is milk cheapest today?' or 'Compare milk prices'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Product name to look up",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Get daily price history for a product over the last N days. "
                "Use this to answer trend questions like 'Is this the cheapest this month?' "
                "or 'Has the price of bread been rising?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Product name"},
                    "days": {
                        "type": "integer",
                        "description": "How many days of history to retrieve (default 14)",
                        "default": 14,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_deals",
            "description": (
                "Return today's biggest price drops (>= 5% off). "
                "Use for questions like 'What are the best deals today?' or 'Any good discounts?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of deals to return (default 10)",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cheapest_by_market",
            "description": (
                "For a product, return one row per market showing the cheapest current price, "
                "ranked cheapest-first. Best for 'Which market has the cheapest eggs?' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Product name"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_for_recipe",
            "description": (
                "Given a recipe or dish name (e.g. 'cake', 'kek', 'pizza', 'börek'), "
                "returns the required ingredients and the cheapest available price for each "
                "from the database. Use this for questions like 'What do I need to make a cake?' "
                "or 'Suggest ingredients for pasta' or 'kek için ne lazım?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe": {
                        "type": "string",
                        "description": "The dish or recipe name (e.g. 'cake', 'kek', 'pizza', 'börek', 'makarna')",
                    },
                },
                "required": ["recipe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_trend_summary",
            "description": (
                "Return min/max/avg price and whether today's price is at its lowest or highest "
                "over the last 30 days. Essential for buy/wait advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Product name"},
                    "days": {
                        "type": "integer",
                        "description": "Days of history to analyse (default 14)",
                        "default": 14,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, args: dict) -> str:
    """Call the matching chatbot_tools function and return a JSON string."""
    try:
        if name == "get_current_prices":
            result = tools.get_current_prices(supabase, **args)
        elif name == "get_price_history":
            result = tools.get_price_history(supabase, **args)
        elif name == "get_best_deals":
            result = tools.get_best_deals(supabase, **args)
        elif name == "get_cheapest_by_market":
            result = tools.get_cheapest_by_market(supabase, **args)
        elif name == "get_price_trend_summary":
            result = tools.get_price_trend_summary(supabase, **args)
        elif name == "suggest_for_recipe":
            result = tools.suggest_for_recipe(supabase, **args)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"Tool {name} failed: {exc}")
        return json.dumps({"error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────────────────────

_TURKISH_CHARS = set("çğışöüÇĞİŞÖÜ")
_TURKISH_WORDS = {"nerede", "fiyat", "bugün", "ucuz", "pahalı", "en", "hangi", "var", "mı", "ne", "kadar"}

def _detect_language(text: str) -> str:
    """Return 'Turkish' if the text looks Turkish, otherwise 'English'."""
    if any(c in _TURKISH_CHARS for c in text):
        return "Turkish"
    words = set(text.lower().split())
    if words & _TURKISH_WORDS:
        return "Turkish"
    return "English"


# ─────────────────────────────────────────────────────────────────────────────
# Core chat function (agentic loop)
# ─────────────────────────────────────────────────────────────────────────────

def _run_agent(session_id: str, user_message: str) -> str:
    """
    Run the agentic loop for one user turn:
      1. Add user message to history
      2. Call OpenAI with tools enabled
      3. If the model calls tools, execute them and loop
      4. Return the final text reply
    """
    history = _history[session_id]

    # Trim history to avoid token bloat
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        _history[session_id] = history

    # Detect language and inject as hard instruction so model cannot ignore it
    lang = _detect_language(user_message)
    wrapped = f"[REPLY IN {lang.upper()} ONLY] {user_message}"

    # Add new user message
    history.append({"role": "user", "content": wrapped})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # Agentic loop — model may call multiple tools in sequence
    for _ in range(3):   # max 3 tool-call rounds
        response = openai_client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # If the model wants to call tools
        if msg.tool_calls:
            # Append assistant's tool-call request to messages
            messages.append(msg)

            # Execute each requested tool
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                logger.info(f"Tool call: {tc.function.name}({args})")
                tool_result = _dispatch_tool(tc.function.name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
            # Loop: let the model see the tool results and decide what to say
            continue

        # No more tool calls — model has written its final reply
        reply = msg.content or ""
        history.append({"role": "assistant", "content": reply})
        return reply

    # Fallback if loop limit hit
    return "I had trouble processing that request. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# Flask Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return jsonify({
        "service": "Pro-Price Assistant",
        "status": "ok",
        "endpoints": {
            "POST /chat":   "Send a message { message, session_id }",
            "DELETE /chat": "Clear history  { session_id }",
            "GET /health":  "Health check",
            "GET /deals":   "Today's best deals",
        }
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Pro-Price Assistant"})


@app.post("/chat")
def chat():
    """
    Request body:
      { "message": "Where is milk cheapest?", "session_id": "user-123" }

    Response:
      { "reply": "At BİM for 24.90 TL 🏆 ..." }
    """
    body = request.get_json(silent=True) or {}
    user_message: Optional[str] = body.get("message", "").strip()
    session_id: str = body.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        reply = _run_agent(session_id, user_message)
        return jsonify({"reply": reply, "session_id": session_id})
    except Exception as exc:
        logger.error(f"Chat error: {exc}", exc_info=True)
        return jsonify({"error": "Internal error, please try again."}), 500


@app.delete("/chat")
def clear_history():
    """
    Clear conversation memory for a session.
    Request body: { "session_id": "user-123" }
    """
    body = request.get_json(silent=True) or {}
    session_id: str = body.get("session_id", "default")
    _history.pop(session_id, None)
    return jsonify({"status": "cleared", "session_id": session_id})


@app.get("/debug")
def debug():
    """Raw Supabase test — bypasses LLM, shows exactly what the DB returns."""
    try:
        resp = supabase.table("price_history").select("product_name, market_name, current_price, scraped_date").limit(5).execute()
        return jsonify({
            "row_count": len(resp.data),
            "rows": resp.data,
            "note": "If row_count is 0 but data exists in Supabase, your API key is blocked by RLS."
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/deals")
def best_deals():
    """
    Raw best-deals endpoint — no LLM involved, just direct Supabase data.
    Query param: limit (default 10)
    """
    limit = min(int(request.args.get("limit", 10)), 50)
    deals = tools.get_best_deals(supabase, limit=limit)
    return jsonify({"deals": deals, "count": len(deals)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Pro-Price Assistant on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
