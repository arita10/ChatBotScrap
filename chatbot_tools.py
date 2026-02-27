"""
chatbot_tools.py — Direct Supabase query helpers for the Price Intelligence Chatbot.

These functions are called by the Flask backend to give the LLM structured,
reliable data instead of letting it write ad-hoc SQL every time.

Functions:
  search_products(supabase, keyword, limit)   — FTS / ILIKE search
  get_current_prices(supabase, keyword)       — all markets for a product today
  get_price_history(supabase, keyword, days)  — recent daily prices for a product
  get_best_deals(supabase, limit)             — today's biggest drops
  get_cheapest_by_market(supabase, keyword)   — rank markets cheapest-first
  get_price_trend_summary(supabase, keyword)  — min/max/avg over last 30 days
"""

import logging
from datetime import date, timedelta
from typing import Optional

from supabase import Client

logger = logging.getLogger("price_chatbot.tools")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()

def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Tool functions
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Recipe → ingredient mapping (Turkish keywords that match DB product names)
# ─────────────────────────────────────────────────────────────────────────────

RECIPE_INGREDIENTS: dict[str, list[str]] = {
    # Baked goods
    "kek":          ["un", "yumurta", "süt", "tereyağ", "şeker", "kabartma"],
    "pasta":        ["un", "yumurta", "süt", "tereyağ", "şeker", "krema", "çikolata"],
    "ekmek":        ["un", "maya", "tuz", "yağ"],
    "kurabiye":     ["un", "yumurta", "tereyağ", "şeker", "vanilya"],
    "börek":        ["yufka", "yumurta", "peynir", "maydanoz", "sıvıyağ"],
    "pizza":        ["un", "maya", "domates", "peynir", "zeytin"],
    # Main dishes
    "tavuk":        ["tavuk", "domates", "soğan", "sarımsak", "biber", "yağ"],
    "köfte":        ["kıyma", "soğan", "yumurta", "ekmek", "tuz", "karabiber"],
    "makarna":      ["makarna", "domates", "peynir", "zeytinyağ", "sarımsak"],
    "pilav":        ["pirinç", "tereyağ", "tuz", "su"],
    "mercimek":     ["mercimek", "soğan", "domates", "havuç", "yağ"],
    "omlet":        ["yumurta", "süt", "tereyağ", "tuz", "kaşar"],
    # Desserts
    "sütlaç":       ["süt", "pirinç", "şeker", "vanilya"],
    "muhallebi":    ["süt", "nişasta", "şeker", "vanilya"],
    "baklava":      ["yufka", "ceviz", "tereyağ", "şeker"],
    # Drinks / breakfast
    "smoothie":     ["süt", "muz", "yoğurt"],
    "kahvaltı":     ["yumurta", "peynir", "ekmek", "domates", "salatalık", "zeytin"],
    # English aliases
    "cake":         ["un", "yumurta", "süt", "tereyağ", "şeker", "kabartma"],
    "bread":        ["un", "maya", "tuz"],
    "omelette":     ["yumurta", "süt", "tereyağ"],
    "pasta dish":   ["makarna", "domates", "peynir"],
    "rice":         ["pirinç", "tereyağ"],
    "cookies":      ["un", "yumurta", "tereyağ", "şeker"],
    "baklava":      ["yufka", "ceviz", "tereyağ", "şeker"],
}


def suggest_for_recipe(supabase: Client, recipe: str) -> dict:
    """
    Given a recipe/dish name, find its ingredients and look up the cheapest
    available price for each ingredient in the DB.
    Returns: { recipe, ingredients: [ {ingredient, found, product_name, market, price} ] }
    """
    key = recipe.lower().strip()

    # Find the best matching recipe key
    ingredients: list[str] = []
    for rkey, ings in RECIPE_INGREDIENTS.items():
        if rkey in key or key in rkey:
            ingredients = ings
            break

    if not ingredients:
        return {"recipe": recipe, "found": False, "ingredients": []}

    results = []
    for ing in ingredients:
        # Get cheapest available product matching this ingredient
        try:
            date_resp = (
                supabase.table("price_history")
                .select("scraped_date")
                .ilike("product_name", f"%{ing}%")
                .order("scraped_date", desc=True)
                .limit(1)
                .execute()
            )
            if not date_resp.data:
                results.append({"ingredient": ing, "found": False})
                continue

            latest_date = date_resp.data[0]["scraped_date"]
            resp = (
                supabase.table("price_history")
                .select("product_name, market_name, current_price")
                .ilike("product_name", f"%{ing}%")
                .eq("scraped_date", latest_date)
                .order("current_price", desc=False)
                .limit(1)
                .execute()
            )
            if resp.data:
                row = resp.data[0]
                results.append({
                    "ingredient": ing,
                    "found": True,
                    "product_name": row["product_name"],
                    "market_name": row["market_name"],
                    "price": row["current_price"],
                })
            else:
                results.append({"ingredient": ing, "found": False})
        except Exception as exc:
            logger.error(f"suggest_for_recipe error for {ing}: {exc}")
            results.append({"ingredient": ing, "found": False})

    found_count = sum(1 for r in results if r.get("found"))
    return {
        "recipe": recipe,
        "found": True,
        "ingredient_count": len(ingredients),
        "found_in_db": found_count,
        "ingredients": results,
    }


def search_products(supabase: Client, keyword: str, limit: int = 20) -> list[dict]:
    """
    Search the products table by name using ILIKE.
    Returns id, product_name, market_name, latest_price, product_url.
    """
    try:
        resp = (
            supabase.table("products")
            .select("product_name, market_name, latest_price, product_url")
            .ilike("product_name", f"%{keyword}%")
            .order("latest_price", desc=False)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"search_products error: {exc}")
        return []


def get_current_prices(supabase: Client, keyword: str) -> list[dict]:
    """
    Return the most recent price records matching keyword across all markets,
    sorted cheapest-first. Uses the latest available scraped_date (not strictly
    today) so data is always returned even if the scraper hasn't run yet today.
    """
    try:
        # Step 1: find the most recent scraped_date for this keyword
        date_resp = (
            supabase.table("price_history")
            .select("scraped_date")
            .ilike("product_name", f"%{keyword}%")
            .order("scraped_date", desc=True)
            .limit(1)
            .execute()
        )
        if not date_resp.data:
            return []
        latest_date = date_resp.data[0]["scraped_date"]

        # Step 2: fetch all records for that date
        resp = (
            supabase.table("price_history")
            .select("product_name, market_name, current_price, previous_price, price_drop_pct, scraped_date, product_url")
            .ilike("product_name", f"%{keyword}%")
            .eq("scraped_date", latest_date)
            .order("current_price", desc=False)
            .limit(30)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"get_current_prices error: {exc}")
        return []


def get_price_history(supabase: Client, keyword: str, days: int = 14) -> list[dict]:
    """
    Return daily price records for a product over the last `days` days.
    Useful for trend analysis (e.g. "is this the lowest price in a month?").
    """
    try:
        cutoff = _days_ago(days)
        resp = (
            supabase.table("price_history")
            .select("product_name, market_name, current_price, previous_price, price_drop_pct, scraped_date")
            .ilike("product_name", f"%{keyword}%")
            .gte("scraped_date", cutoff)
            .order("scraped_date", desc=True)
            .limit(days * 5)   # up to 5 markets × days
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"get_price_history error: {exc}")
        return []


def get_best_deals(supabase: Client, limit: int = 10) -> list[dict]:
    """
    Today's biggest price drops (price_drop_pct >= 5%).
    Tries v_best_deals view first, falls back to direct query.
    """
    try:
        resp = (
            supabase.table("v_best_deals")
            .select("product_name, market_name, current_price, previous_price, price_drop_pct, product_url")
            .limit(limit)
            .execute()
        )
        if resp.data:
            return resp.data
    except Exception:
        pass

    # Fallback: use the most recent available date
    try:
        date_resp = (
            supabase.table("price_history")
            .select("scraped_date")
            .order("scraped_date", desc=True)
            .limit(1)
            .execute()
        )
        latest_date = date_resp.data[0]["scraped_date"] if date_resp.data else _today()
        resp = (
            supabase.table("price_history")
            .select("product_name, market_name, current_price, previous_price, price_drop_pct, product_url")
            .eq("scraped_date", latest_date)
            .gte("price_drop_pct", 5)
            .order("price_drop_pct", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"get_best_deals fallback error: {exc}")
        return []


def get_cheapest_by_market(supabase: Client, keyword: str) -> list[dict]:
    """
    For a given product keyword, return one row per market showing the
    cheapest current price, ranked cheapest-first.
    """
    rows = get_current_prices(supabase, keyword)
    if not rows:
        return []

    # Deduplicate: keep only the cheapest row per market
    seen: dict[str, dict] = {}
    for row in rows:
        mkt = row["market_name"]
        if mkt not in seen or row["current_price"] < seen[mkt]["current_price"]:
            seen[mkt] = row

    return sorted(seen.values(), key=lambda r: r["current_price"])


def get_price_trend_summary(supabase: Client, keyword: str, days: int = 14) -> dict:
    """
    Return min/max/avg price + current price for a product over `days` days.
    Used to generate "lowest in X days!" or "currently above average" advice.
    """
    history = get_price_history(supabase, keyword, days)
    if not history:
        return {}

    prices = [r["current_price"] for r in history if r.get("current_price") is not None]
    if not prices:
        return {}

    # Most recent record
    latest = history[0]

    lo, hi = min(prices), max(prices)
    return {
        "product_name": latest.get("product_name", keyword),
        "market_name": latest.get("market_name"),
        "current_price": latest.get("current_price"),
        "min_price": lo,
        "max_price": hi,
        "avg_price": round(sum(prices) / len(prices), 2),
        "is_at_lowest": latest.get("current_price") == lo,
        "is_at_highest": latest.get("current_price") == hi,
        "price_drop_pct": latest.get("price_drop_pct"),
    }
