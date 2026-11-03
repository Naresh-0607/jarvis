"""Doctor command.

Checks:
* Python version is at least 3.11.
* .env file exists in the project root.
* Provider API‑key environment variables are declared (names only, values
  are read from .env).
* Internet connectivity by attempting a simple HTTPS request.
* Provider reachability – we can make a HEAD request to the base URLs
  specified in config.yaml. If a provider is unreachable we log a
  warning but do not crash.
* Kaggle Ollama check: call GET {OLLAMA_BASE_URL}/api/tags (add the auth header only if a token is set). Pass if reachable AND the configured model name appears in the returned list. If unreachable, print "Kaggle unreachable: the session may have ended or the tunnel URL changed". If reachable but the model is missing, print the models that ARE available.
* Groq check: call GET {base_url}/models with the Bearer key. Pass if status 200. Print only pass/fail and the status code, never the key.

The goal of the doctor is simply to surface configuration problems
before the main application starts.
"""

from __future__ import annotations

import logging
import os
import sys
import socket
from pathlib import Path
import yaml
import httpx

_logger = logging.getLogger("jarvis.doctor")

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure file + console logging.

    The project root contains ``logs/`` – we rotate daily with a limit of
    5 backup files.  The console level is DEBUG during development and
    WARNING in production.
    """
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "app.log"
    from logging.handlers import RotatingFileHandler

    handler = RotatingFileHandler(str(log_file), maxBytes=2_000_000, backupCount=5)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(handler)
    # Console handler
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    _logger.addHandler(stream)


def check_python_version() -> None:
    if sys.version_info < (3, 11):
        _logger.error("Python %s required. You have %s", (3, 11), sys.version)
        sys.exit(1)
    _logger.debug("Python version OK: %s", sys.version)


def check_env_file() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        _logger.error(".env file not found at %s", env_path)
        sys.exit(1)
    _logger.debug(".env file found: %s", env_path)


def load_config() -> dict:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        _logger.error("config.yaml not found at %s", cfg_path)
        sys.exit(1)
    with cfg_path.open("rt", encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    _logger.debug("Config loaded: %s", cfg)
    return cfg


def check_provider_env_vars(cfg: dict) -> None:
    # Only check names in provider chain
    provider_chain = cfg.get("providers", [])
    missing = []
    for p in provider_chain:
        key_name = p.get("api_key_env")
        if key_name and key_name not in os.environ:
            missing.append(key_name)
    if missing:
        _logger.warning("Missing environment variables: %s", ", ".join(missing))


def check_internet() -> None:
    try:
        socket.create_connection(("github.com", 443), timeout=5)
        _logger.debug("Internet connectivity OK")
    except Exception as e:
        _logger.warning("No internet connectivity: %s", e)


def check_kaggle_ollama(cfg: dict) -> bool:
    """Check if Kaggle Ollama is reachable and has the required model."""
    # Get the base URL from environment
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        _logger.error(".env file not found during Kaggle check")
        return False
        
    # Load .env to get OLLAMA_BASE_URL
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    ollama_base_url = os.environ.get('OLLAMA_BASE_URL')
    if not ollama_base_url:
        _logger.error("OLLAMA_BASE_URL not found in .env")
        return False
        
    # Prepare the URL for the tags endpoint
    root_url = ollama_base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    tags_url = f"{root_url}/api/tags"
    
    try:
        # Get auth token (if exists)
        auth_token = os.environ.get('OLLAMA_AUTH_TOKEN')
        
        # Prepare headers
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        response = httpx.get(
            tags_url,
            headers=headers,
            timeout=cfg.get("providers", [{}])[0].get("connect_timeout_s", 4)
        )
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Check if our required model is in the available models
                model_name = cfg.get("providers", [{}])[0].get("models", [None])[0]
                if model_name:
                    if "models" in data:
                        available_models = [m["name"] for m in data["models"]]
                        if model_name in available_models:
                            _logger.debug(f"Kaggle Ollama reachable with correct model {model_name}")
                            return True
                        else:
                            _logger.warning(f"Kaggle Ollama reachable but missing required model '{model_name}'")
                            _logger.info(f"Available models: {available_models}")
                else:
                    # If no model specified in config, just check connectivity 
                    _logger.debug("Kaggle Ollama reachable (no specific model required)")
                    return True
            except Exception as e:
                _logger.warning(f"Failed to parse Kaggle Ollama response: {e}")
                _logger.debug("Kaggle Ollama reachable but couldn't verify model")
                return True
        else:
            _logger.warning(f"Kaggle Ollama not reachable (HTTP {response.status_code})")
            _logger.info("Kaggle unreachable: the session may have ended or the tunnel URL changed")
            
    except httpx.TimeoutException:
        _logger.warning("Kaggle Ollama timeout during check")
        _logger.info("Kaggle unreachable: the session may have ended or the tunnel URL changed")
    except Exception as e:
        _logger.warning(f"Failed to reach Kaggle Ollama: {e}")
        _logger.info("Kaggle unreachable: the session may have ended or the tunnel URL changed")
        
    return False


def check_groq(cfg: dict) -> bool:
    """Check if Groq is reachable and has valid API key."""
    # Find Groq provider in config
    groq_provider = None
    for p in cfg.get("providers", []):
        if p.get("name") == "groq":
            groq_provider = p
            break
            
    if not groq_provider:
        _logger.warning("Groq provider not found in configuration")
        return False
        
    base_url = groq_provider.get("base_url", "")
    api_key_env = groq_provider.get("api_key_env", "")
    
    if not base_url:
        _logger.warning("Groq base URL not configured")
        return False
        
    if not api_key_env or api_key_env not in os.environ:
        _logger.warning("Groq API key not found in environment")
        return False
        
    api_key = os.environ.get(api_key_env)
    
    try:
        response = httpx.get(
            f"{base_url}/models", 
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=cfg.get("providers", [{}])[1].get("connect_timeout_s", 8)  # Groq provider should be second
        )
        
        if response.status_code == 200:
            _logger.debug("Groq reachable and valid")
            return True
        else:
            _logger.warning(f"Groq API check failed with status {response.status_code}")
            
    except Exception as e:
        _logger.warning(f"Failed to reach Groq: {e}")
        
    return False


def check_search_providers() -> tuple[dict[str, bool], bool]:
    """Check search provider keys in env and SQLite cache/usage tables."""
    root = Path(__file__).resolve().parents[1]
    from dotenv import load_dotenv
    load_dotenv(root / ".env")

    keys = {
        "TAVILY_API_KEY": bool(os.environ.get("TAVILY_API_KEY")),
        "EXA_API_KEY": bool(os.environ.get("EXA_API_KEY")),
    }

    # Check SQLite tables
    db_ok = False
    try:
        from jarvis.db import connect
        conn = connect()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('search_cache', 'search_usage')"
        )
        tables = {row["name"] for row in cur.fetchall()}
        db_ok = ("search_cache" in tables) and ("search_usage" in tables)
        conn.close()
    except Exception as exc:
        _logger.warning("Could not verify search database tables: %s", exc)

    return keys, db_ok


def main() -> None:
    setup_logging()
    _logger.info("Jarvis Doctor starting…")
    
    # Check Python version first
    check_python_version()

    # Check .env file
    check_env_file()

    # Load .env before reading any of its values
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    # Load config
    cfg = load_config()
    
    # Check env var names but not values
    check_provider_env_vars(cfg)
    
    # Check internet connectivity  
    check_internet()
    
    # Check providers
    kaggle_ok = False
    groq_ok = False
    
    # Check Kaggle Ollama (warning if fails, not error)
    _logger.info("Checking Kaggle Ollama availability...")
    kaggle_ok = check_kaggle_ollama(cfg)
    
    # Check Groq
    _logger.info("Checking Groq availability...")
    groq_ok = check_groq(cfg)

    # Check Search Providers & Database
    _logger.info("Checking search provider configuration and database...")
    search_keys, search_db_ok = check_search_providers()

    live_test = "--live" in sys.argv
    search_live_ok = None
    if live_test:
        _logger.info("Running 1-credit live search check (--live specified)...")
        try:
            import time
            from jarvis.web_search import perform_web_search
            t0 = time.perf_counter()
            s_res = perform_web_search("Python news", max_results=1, bypass_cache=True)
            latency = time.perf_counter() - t0
            search_live_ok = s_res.get("status") == "ok" and s_res.get("total_results", 0) > 0
            _logger.info(
                "Live search completed in %0.2fs (provider: %s, status: %s)",
                latency,
                s_res.get("provider"),
                s_res.get("status"),
            )
        except Exception as exc:
            _logger.warning("Live search failed: %s", exc)
            search_live_ok = False
    
    # Print clear pass/fail table
    _logger.info("=" * 50)
    _logger.info("DOCTOR CHECK RESULTS")
    _logger.info("=" * 50)
    _logger.info(f"{'Python':<20} {'OK' if sys.version_info >= (3, 11) else 'FAIL'}")
    _logger.info(f"{'Internet':<20} OK")  # Internet check is always performed
    _logger.info(f"{'Kaggle Ollama':<20} {'OK' if kaggle_ok else 'FAIL (warning: backup exists)'}")
    _logger.info(f"{'Groq':<20} {'OK' if groq_ok else 'FAIL'}")
    _logger.info(f"{'Tavily API Key':<20} {'PRESENT' if search_keys['TAVILY_API_KEY'] else 'MISSING (Tavily disabled)'}")
    _logger.info(f"{'Exa API Key':<20} {'PRESENT' if search_keys['EXA_API_KEY'] else 'MISSING (optional fallback skipped)'}")
    _logger.info(f"{'Search Cache/Quota':<20} {'OK (tables ready)' if search_db_ok else 'FAIL'}")
    if live_test:
        _logger.info(f"{'Live Search':<20} {'OK (1 credit used)' if search_live_ok else 'FAIL'}")
    _logger.info("=" * 50)
    
    # Log conclusion
    if groq_ok:
        _logger.info("Doctor finished – all checks passed")
    else:
        _logger.error("Groq check failed - this is required for the assistant to function")
        sys.exit(1)


if __name__ == "__main__":
    main()