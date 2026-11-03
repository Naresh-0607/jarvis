"""Unit tests for Phase 8 - General Tools (Weather, Web Search, Time, Math, Router)."""

from datetime import datetime
import pytest
from unittest.mock import MagicMock, patch

from jarvis.general_tools import calculate, current_time, get_weather, web_search
from jarvis.router import FAST, STRONG, choose_tier


# ---------------------------------------------------------------------------
# Current Time Tests
# ---------------------------------------------------------------------------

def test_current_time_default():
    res = current_time()
    assert "time" in res
    assert "date" in res
    assert "day_of_week" in res
    assert res["timezone"] == "Asia/Kolkata"
    # Validate ISO format
    dt = datetime.fromisoformat(res["iso"])
    assert dt is not None


def test_current_time_custom_timezone():
    res = current_time(timezone="UTC")
    assert res["timezone"] == "UTC"
    assert res["utc_offset"] == "+0000"


# ---------------------------------------------------------------------------
# Safe Calculator Tests
# ---------------------------------------------------------------------------

def test_calculate_basic_arithmetic():
    assert calculate("2 + 2")["result"] == 4
    assert calculate("10 * 5 - 8 / 2")["result"] == 46
    assert calculate("100 // 3")["result"] == 33
    assert calculate("10 % 3")["result"] == 1
    assert calculate("2 ** 8")["result"] == 256


def test_calculate_math_functions():
    assert calculate("sqrt(144)")["result"] == 12
    assert calculate("abs(-42)")["result"] == 42
    assert calculate("round(3.14159, 2)")["result"] == 3.14
    assert calculate("max(10, 20, 5)")["result"] == 20
    assert calculate("min(10, 20, 5)")["result"] == 5


def test_calculate_rejects_unsafe_operations():
    # Attempting to call unauthorized functions
    assert "error" in calculate("__import__('os').system('dir')")
    assert "error" in calculate("open('config.yaml').read()")
    assert "error" in calculate("exec('print(1)')")
    # Massive exponent protection
    assert "error" in calculate("2 ** 50000")


# ---------------------------------------------------------------------------
# Weather Tool Tests (with mock)
# ---------------------------------------------------------------------------

@patch("httpx.Client.get")
def test_get_weather_success(mock_get):
    # Geocoding mock
    geo_mock = MagicMock()
    geo_mock.status_code = 200
    geo_mock.json.return_value = {
        "results": [{"name": "New Delhi", "country": "India", "latitude": 28.6, "longitude": 77.2}]
    }

    # Forecast mock
    forecast_mock = MagicMock()
    forecast_mock.status_code = 200
    forecast_mock.json.return_value = {
        "current": {
            "temperature_2m": 26.5,
            "apparent_temperature": 27.0,
            "relative_humidity_2m": 60,
            "weather_code": 0,
            "wind_speed_10m": 12.0,
        },
        "daily": {
            "temperature_2m_max": [31.0, 30.5],
            "temperature_2m_min": [19.0, 18.5],
            "weather_code": [0, 1],
            "precipitation_probability_max": [5, 10],
        },
    }

    mock_get.side_effect = [geo_mock, forecast_mock]

    res = get_weather("New Delhi")
    assert "error" not in res
    assert "New Delhi, India" in res["city"]
    assert res["now"]["temperature_c"] == 26.5
    assert res["now"]["condition"] == "Clear skies"
    assert res["today"]["high_c"] == 31.0
    assert res["today"]["rain_chance_percent"] == 5


@patch("httpx.Client.get")
def test_get_weather_city_not_found(mock_get):
    geo_mock = MagicMock()
    geo_mock.status_code = 200
    geo_mock.json.return_value = {"results": []}
    mock_get.return_value = geo_mock

    res = get_weather("NonexistentAtlantisCity12345")
    assert "error" in res
    assert "not found" in res["error"].lower()


# ---------------------------------------------------------------------------
# Web Search Tests (with mock)
# ---------------------------------------------------------------------------

@patch("ddgs.DDGS")
def test_web_search_success(mock_ddgs_cls):
    mock_inst = MagicMock()
    mock_inst.text.return_value = [
        {
            "title": "Python 3.12 Release Notes",
            "body": "Python 3.12 introduces new syntax features and performance improvements.",
            "href": "https://docs.python.org/3/whatsnew/3.12.html",
        }
    ]
    mock_ddgs_cls.return_value = mock_inst

    res = web_search("python release", max_results=1)
    assert res["total_results"] == 1
    assert "Python 3.12" in res["results"][0]["title"]
    assert "docs.python.org" in res["results"][0]["source"]


# ---------------------------------------------------------------------------
# Router Improvements Tests
# ---------------------------------------------------------------------------

def test_router_routes_lookups_to_fast():
    assert choose_tier("what is the weather in Delhi?") == FAST
    assert choose_tier("what's the temperature outside?") == FAST
    assert choose_tier("what time is it right now?") == FAST
    assert choose_tier("calculate 15 * 84") == FAST
    assert choose_tier("remind me to buy milk tomorrow at 5") == FAST
    assert choose_tier("hello jarvis") == FAST


def test_router_routes_complex_reasoning_to_strong():
    assert choose_tier("why does this error happen?") == STRONG
    assert choose_tier("explain how the event loop works step by step") == STRONG
    assert choose_tier("teach me how to use FreeRTOS on ESP32") == STRONG
    assert choose_tier("debug this traceback: KeyError: 'user_id'") == STRONG
    assert choose_tier("compare monolith vs microservices architecture") == STRONG
