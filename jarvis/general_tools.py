"""General-purpose tools for weather, web search, time, and safe calculation (Phase 8).

Tools:
* ``get_weather``: Open-Meteo geocoding and forecast API (no API key required).
* ``web_search``: DuckDuckGo search returning top titles, snippets, and source URLs.
* ``current_time``: accurate date and time formatting in project or custom timezone.
* ``calculate``: safe AST-based arithmetic and math evaluation (no code injection).
"""

from __future__ import annotations

import ast
from datetime import datetime
import logging
import math
import operator
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo
import yaml

import httpx

from jarvis.db import PROJECT_ROOT
from jarvis.tools import tool

_logger = logging.getLogger("jarvis.general_tools")

# ---------------------------------------------------------------------------
# Weather Code Descriptions (WMO Code mapping)
# ---------------------------------------------------------------------------
WMO_WEATHER_CODES = {
    0: "Clear skies",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _get_home_city() -> str:
    """Read home city from config.yaml, defaulting to 'New Delhi'."""
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        try:
            with cfg_path.open("rt", encoding="utf8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("home_city", "New Delhi")
        except Exception:
            pass
    return "New Delhi"


def _get_default_timezone() -> str:
    """Read timezone from config.yaml, defaulting to 'Asia/Kolkata'."""
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        try:
            with cfg_path.open("rt", encoding="utf8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("timezone", "Asia/Kolkata")
        except Exception:
            pass
    return "Asia/Kolkata"


# ---------------------------------------------------------------------------
# Weather Tool (Open-Meteo)
# ---------------------------------------------------------------------------

WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": (
        "Get live weather forecast for a city using Open-Meteo. "
        "Returns current temperature, conditions, today's high/low, rain probability, "
        "and tomorrow's forecast. Leave city empty to use home city."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g. 'London', 'Tokyo', 'Mumbai'). Default: home city.",
                "default": "",
            },
        },
    },
}


@tool(name="get_weather", schema=WEATHER_SCHEMA)
def get_weather(city: str = "") -> Dict[str, Any]:
    """Fetch live weather from Open-Meteo API."""
    target_city = (city or "").strip() or _get_home_city()
    timeout_s = 6.0

    try:
        # Step 1: Geocoding lookup
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": target_city, "count": 1, "language": "en", "format": "json"}
        with httpx.Client(timeout=timeout_s) as client:
            geo_res = client.get(geo_url, params=geo_params)
            if geo_res.status_code != 200:
                return {
                    "error": f"Could not look up location for '{target_city}' (HTTP {geo_res.status_code})."
                }

            geo_data = geo_res.json()
            results = geo_data.get("results")
            if not results:
                return {"error": f"Location '{target_city}' was not found. Ask the user to clarify the city."}

            loc = results[0]
            lat = loc["latitude"]
            lon = loc["longitude"]
            resolved_name = f"{loc.get('name')}, {loc.get('country')}"

            # Step 2: Forecast lookup
            forecast_url = "https://api.open-meteo.com/v1/forecast"
            forecast_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 2,
            }
            forecast_res = client.get(forecast_url, params=forecast_params)
            if forecast_res.status_code != 200:
                return {
                    "error": f"Weather forecast unavailable for {resolved_name} (HTTP {forecast_res.status_code})."
                }

            fc_data = forecast_res.json()

        current = fc_data.get("current", {})
        daily = fc_data.get("daily", {})

        current_code = current.get("weather_code", 0)
        current_condition = WMO_WEATHER_CODES.get(current_code, "Unknown")
        current_temp = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind_speed = current.get("wind_speed_10m")

        # Today's daily values
        today_high = daily.get("temperature_2m_max", [None])[0]
        today_low = daily.get("temperature_2m_min", [None])[0]
        today_rain_chance = daily.get("precipitation_probability_max", [None])[0]

        # Tomorrow's daily values
        tomorrow_high = daily.get("temperature_2m_max", [None, None])[1] if len(daily.get("temperature_2m_max", [])) > 1 else None
        tomorrow_low = daily.get("temperature_2m_min", [None, None])[1] if len(daily.get("temperature_2m_min", [])) > 1 else None
        tomorrow_code = daily.get("weather_code", [None, None])[1] if len(daily.get("weather_code", [])) > 1 else None
        tomorrow_condition = WMO_WEATHER_CODES.get(tomorrow_code, "Fair") if tomorrow_code is not None else None
        tomorrow_rain_chance = daily.get("precipitation_probability_max", [None, None])[1] if len(daily.get("precipitation_probability_max", [])) > 1 else None

        return {
            "city": resolved_name,
            "now": {
                "temperature_c": current_temp,
                "condition": current_condition,
                "feels_like_c": feels_like,
                "humidity_percent": humidity,
                "wind_kmh": wind_speed,
            },
            "today": {
                "high_c": today_high,
                "low_c": today_low,
                "rain_chance_percent": today_rain_chance,
            },
            "tomorrow": {
                "condition": tomorrow_condition,
                "high_c": tomorrow_high,
                "low_c": tomorrow_low,
                "rain_chance_percent": tomorrow_rain_chance,
            },
        }

    except httpx.TimeoutException:
        _logger.warning("Weather API request timed out for '%s'", target_city)
        return {"error": "Weather service request timed out. Mention that you couldn't check the weather."}
    except Exception as exc:
        _logger.error("Failed to fetch weather for '%s': %s", target_city, exc)
        return {"error": f"Weather lookup failed: {exc}"}


# ---------------------------------------------------------------------------
# Web Search & Quota Tools (Tavily, Exa, DDGS via web_search module)
# ---------------------------------------------------------------------------

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for up-to-date facts, current news, people, and technical documentation. "
        "IMPORTANT: Always pass a FULL, self-contained query combining the user's original topic with any "
        "subsequent hints or context (e.g. if the user asked about 'jk south indian youtuber' and later hinted "
        "'does comedy in Tamil', pass 'JK Tamil comedy YouTuber'). Never search only the hint. "
        "Resolve all pronouns ('he', 'she', 'it', 'that') and expand abbreviations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Full, self-contained search query combining the question and any hints.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to retrieve (default 5, max 10)",
                "default": 5,
            },
            "recent": {
                "type": "boolean",
                "description": "Set to true if searching for live news, breaking events, or recent updates.",
                "default": False,
            },
        },
        "required": ["query"],
    },
}


@tool(name="web_search", schema=WEB_SEARCH_SCHEMA)
def web_search(query: str, max_results: int = 5, recent: bool = False) -> Dict[str, Any]:
    """Execute multi-provider web search pipeline."""
    from jarvis.web_search import perform_web_search
    return perform_web_search(query=query, max_results=max_results, recent=recent)


SEARCH_USAGE_SCHEMA = {
    "name": "search_usage",
    "description": (
        "Check how many web search credits have been used this month and how many are remaining "
        "for Tavily and Exa providers."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


@tool(name="search_usage", schema=SEARCH_USAGE_SCHEMA)
def search_usage() -> Dict[str, Any]:
    """Return credits used and remaining this month per search provider."""
    from jarvis.web_search import get_search_usage_summary
    return get_search_usage_summary()


# ---------------------------------------------------------------------------
# Current Time & Date Tool
# ---------------------------------------------------------------------------

CURRENT_TIME_SCHEMA = {
    "name": "current_time",
    "description": (
        "Get current exact date, time, and day of the week in a specified timezone. "
        "Defaults to project timezone (Asia/Kolkata). Always call this instead of guessing time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional IANA timezone name (e.g. 'UTC', 'Asia/Kolkata', 'America/New_York')",
                "default": "",
            },
        },
    },
}


@tool(name="current_time", schema=CURRENT_TIME_SCHEMA)
def current_time(timezone: str = "") -> Dict[str, Any]:
    """Return structured current time and date."""
    tz_str = (timezone or "").strip() or _get_default_timezone()
    try:
        tz = ZoneInfo(tz_str)
    except Exception:
        tz = ZoneInfo(_get_default_timezone())
        tz_str = _get_default_timezone()

    now_dt = datetime.now(tz)
    return {
        "iso": now_dt.isoformat(timespec="seconds"),
        "time": now_dt.strftime("%I:%M %p"),
        "date": now_dt.strftime("%d %B %Y"),
        "day_of_week": now_dt.strftime("%A"),
        "timezone": tz_str,
        "utc_offset": now_dt.strftime("%z"),
    }


# ---------------------------------------------------------------------------
# Safe Calculator Tool (AST-based)
# ---------------------------------------------------------------------------

CALCULATE_SCHEMA = {
    "name": "calculate",
    "description": (
        "Perform exact mathematical arithmetic or calculations safely (e.g. '24 * 60 * 365', 'sqrt(144) + 15', '2**10'). "
        "Always call this tool for arithmetic, percentages, or conversions rather than guessing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g. '15 * 1.18', 'sqrt(256)', '2**16')",
            },
        },
        "required": ["expression"],
    },
}

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate safe AST math expressions."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_type == ast.Pow and (right > 1000 or left > 10000):
            raise ValueError("Exponent too large to evaluate safely")
        return _SAFE_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call")
        func_name = node.func.id.lower()
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not permitted")
        func = _SAFE_FUNCTIONS[func_name]
        args = [_safe_eval_node(arg) for arg in node.args]
        return func(*args)

    if isinstance(node, ast.Name):
        name = node.id.lower()
        if name in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[name]
            if isinstance(val, (int, float)):
                return val
        raise ValueError(f"Identifier '{node.id}' is not recognized")

    raise ValueError(f"Expression contains unsupported construct: {type(node).__name__}")


@tool(name="calculate", schema=CALCULATE_SCHEMA)
def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate math expressions without eval()."""
    expr = (expression or "").strip()
    if not expr:
        return {"error": "Expression cannot be empty"}

    try:
        parsed = ast.parse(expr, mode="eval")
        result = _safe_eval_node(parsed.body)
        # Format floating numbers nicely
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 6)
        return {"expression": expr, "result": result}
    except Exception as exc:
        _logger.warning("Safe calculation failed for '%s': %s", expr, exc)
        return {"error": f"Could not calculate '{expr}': {exc}"}
