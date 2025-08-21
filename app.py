import asyncio
import json
import logging
import os
import re
import tempfile
import time
import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
try:
    import google.generativeai as genai  # type: ignore
    from google.generativeai.types.generation_types import GenerationConfig  # type: ignore
except Exception:
    # Gemini is optional. When not available, summarization will return empty strings.
    genai = None  # type: ignore
    GenerationConfig = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and configuration
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Configuration files
DATA_DIR = os.getenv("WATCHER_DATA_DIR", ".")
SOURCES_FILE: str = os.path.join(DATA_DIR, "sources.json")
MEMORY_FILE: str = os.path.join(DATA_DIR, "memory_db.json")
RECIPIENTS_FILE: str = os.path.join(DATA_DIR, "recipients.json")

DEFAULT_MEMORY: Dict[str, Any] = {
    "seen_urls": [],
    "details": {},
    "reports": [],
}

# Memory limits
MAX_SEEN_URLS = int(os.getenv("WATCHER_MAX_SEEN_URLS", "10000"))
MAX_REPORTS = int(os.getenv("WATCHER_MAX_REPORTS", "100"))

# User agent for requests
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; WatcherBot/2.0; +https://example.com/bot) "
    "PythonRequests"
)

# Concurrency lock file
LOCK_FILE = os.path.join(DATA_DIR, ".watch_lock")

# ---------------------------------------------------------------------------
# Utility functions

def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication:
    - Force scheme to https
    - Lowercase the scheme and host
    - Remove default port
    - Strip trailing slash
    - Drop common tracking query parameters (utm_*, gclid, ref)
    """
    if not url:
        return ""
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

        parsed = urlparse(url)
        scheme = parsed.scheme.lower() or "http"
        netloc = parsed.netloc.lower()
        # Remove default port
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        if netloc.endswith(":443"):
            netloc = netloc[:-4]
        # Force https
        if scheme != "https":
            scheme = "https"
        path = parsed.path or ""
        # Remove trailing slash (but keep root "/")
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        # Clean query parameters
        params = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            k = key.lower()
            if k.startswith("utm_") or k in ("gclid", "fbclid", "ref"):
                continue
            params.append((key, value))
        query = urlencode(params, doseq=True)
        # Reassemble
        normalized = urlunparse((scheme, netloc, path, "", query, ""))
        return normalized
    except Exception:
        return url.strip()


def safe_load_json(path: str, default: Any) -> Any:
    """
    Safely load JSON from a file. Returns the default if the file doesn't exist
    or is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load JSON from {path}: {e}. Resetting.")
        return default


def atomic_save_json(data: Any, path: str) -> None:
    """
    Save JSON to disk atomically to avoid corruption.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def safe_load_recipients(path: str = RECIPIENTS_FILE) -> List[str]:
    """
    Load the list of recipient email addresses. Invalid entries are ignored.
    """
    data = safe_load_json(path, [])
    recipients: List[str] = []
    for item in data if isinstance(data, list) else []:
        try:
            addr = str(item).strip()
            # Very basic email validation
            if re.match(r"^[^@]+@[^@]+\.[^@]+$", addr):
                recipients.append(addr)
            else:
                logger.warning(f"Ignoring invalid email address in recipients: {addr}")
        except Exception:
            continue
    # Deduplicate
    return sorted(set(recipients))


def atomic_save_recipients(recipients: List[str], path: str = RECIPIENTS_FILE) -> None:
    """
    Save the recipients list atomically.
    """
    atomic_save_json(sorted(set(recipients)), path)


def safe_load_memory(path: str = MEMORY_FILE) -> Dict[str, Any]:
    """
    Load the persistent memory structure safely, ensuring required keys exist.
    """
    data = safe_load_json(path, DEFAULT_MEMORY.copy())
    memory: Dict[str, Any] = {
        "seen_urls": data.get("seen_urls", []),
        "details": data.get("details", {}),
        "reports": data.get("reports", []),
    }
    # Validate types
    if not isinstance(memory["seen_urls"], list):
        logger.warning("'seen_urls' is not a list. Resetting.")
        memory["seen_urls"] = []
    if not isinstance(memory["details"], dict):
        logger.warning("'details' is not a dict. Resetting.")
        memory["details"] = {}
    if not isinstance(memory["reports"], list):
        logger.warning("'reports' is not a list. Resetting.")
        memory["reports"] = []
    return memory


def atomic_save_memory(memory: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    """
    Save the memory with deduplication and truncation according to limits.
    """
    mem_copy = dict(memory)
    # Deduplicate and truncate seen_urls
    seen_urls = list(dict.fromkeys(mem_copy.get("seen_urls", [])))  # preserves order
    if len(seen_urls) > MAX_SEEN_URLS:
        seen_urls = seen_urls[-MAX_SEEN_URLS:]
    mem_copy["seen_urls"] = seen_urls
    # Truncate reports
    reports = mem_copy.get("reports", [])
    if len(reports) > MAX_REPORTS:
        reports = reports[-MAX_REPORTS:]
    mem_copy["reports"] = reports
    atomic_save_json(mem_copy, path)


# ---------------------------------------------------------------------------
# HTTP and scraping helpers

class HTTPClient:
    """
    A simple HTTP client using requests.Session with retry logic.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        )
        # Configure retries
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str, timeout: int = 10) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=timeout)
            if resp.status_code >= 400:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return None
            # Attempt to set correct encoding
            if resp.encoding is None:
                resp.encoding = resp.apparent_encoding  # type: ignore
            return resp.text
        except Exception as e:
            logger.warning(f"Network error fetching {url}: {e}")
            return None


http_client = HTTPClient()


def _clean_text(text: str) -> str:
    """
    Normalize whitespace and collapse excessive blank lines.
    """
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_main_text(html: str) -> Tuple[str, str]:
    """
    Extract title and main text from an HTML document. If BeautifulSoup is not
    available, fall back to regex tag stripping.
    """
    if not html:
        return ("", "")
    if BeautifulSoup is None:
        title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        title = title_match.group(1).strip() if title_match else ""
        # Strip scripts/styles
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return (title, _clean_text(text))
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts, styles, and non-content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ["nav", "footer", "header", "form", "aside"]:
        for t in soup.select(sel):
            t.decompose()
    # Title extraction
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()  # type: ignore
    # Attempt to find main text inside article or main tags
    candidates = soup.select("article") or soup.select("main") or [soup.body or soup]
    chunks: List[str] = []
    for node in candidates:
        text = node.get_text(separator="\n", strip=True)
        if text:
            chunks.append(text)
    text = "\n\n".join(chunks) if chunks else soup.get_text(separator="\n", strip=True)
    return (title, _clean_text(text))


def fetch_url_text(url: str, timeout: int = 12) -> Tuple[str, str]:
    """
    Fetch an HTML page and extract title and main text. Returns empty strings
    on failure.
    """
    html = http_client.get(url, timeout=timeout)
    if not html:
        return ("", "")
    return extract_main_text(html)


# ---------------------------------------------------------------------------
# LLM helpers

def call_gemini_with_retry(
    prompt: str,
    max_retries: int = 2,
    initial_delay: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Call the Gemini API with retry logic and fall back to alternative API keys.
    Returns the response text or an empty string on failure.
    """
    # If Gemini SDK is unavailable, return empty string
    if genai is None or GenerationConfig is None:
        logger.warning("google.generativeai is not installed. Skipping Gemini summarization.")
        return ""
    api_keys = os.environ.get("GEMINI_API_KEY", "").split(",")
    if not api_keys or not api_keys[0]:
        logger.error("GEMINI_API_KEY is not configured.")
        return ""
    last_error: Optional[Exception] = None
    for key in api_keys:
        key = key.strip()
        if not key:
            continue
        try:
            genai.configure(api_key=key)
            # attempt call with single key
            tools = [
              {"url_context": {}},
              {"google_search": {}}
            ]
            model = genai.GenerativeModel(model_name=model_name)
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(
                        contents=[
                            {"role": "user", "content": prompt}
                        ],
                        tools=tools,
                        generation_config=GenerationConfig()
                    )
                    # response = model.generate_content(
                    #     contents=[{"role": "user", "content": prompt}],
                    #     generation_config=GenerationConfig(tools=tools),
                    # )
                    if hasattr(response, "text") and response.text:
                        return response.text.strip()
                    if getattr(response, "candidates", None):
                        candidate = response.candidates[0]
                        text = getattr(candidate, "text", "") or getattr(candidate, "content", "")
                        if text:
                            return str(text).strip()
                    logger.warning(
                        f"Empty response from Gemini (attempt {attempt + 1}/{max_retries})"
                    )
                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Gemini error (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                # Backoff
                if attempt < max_retries - 1:
                    sleep_time = initial_delay * (2 ** attempt)
                    time.sleep(sleep_time)
        except Exception as e:
            last_error = e
            logger.error(f"Failed to configure Gemini with provided API key: {e}")
    if last_error:
        logger.error(f"All Gemini calls failed. Last error: {last_error}")
    return ""


def summarize_with_url_context(url: str, scraped_text: str) -> str:
    """
    Summarize the content of a URL using Gemini. The URL is passed in the prompt
    to give the model context. Falls back to summarizing the scraped text.
    """
    url_prompt = (
        "Analyse et résume précisément en français le contenu de cette URL. "
        "Mets en avant les mises à jour, nouvelles informations et points clés. "
        "Structure la réponse avec des puces claires et un court paragraphe de synthèse à la fin."
    )
    try:
        combined_prompt = f"{url_prompt}\n\nURL: {url}"
        text = call_gemini_with_retry(prompt=combined_prompt)
        if text:
            return text
    except Exception as e:
        logger.info(f"URL context summarization failed for {url}: {e}")
    if scraped_text:
        fallback_prompt = (
            "Voici le contenu d'une page web. Résume-le en français, en listant d'abord les points clés, "
            "puis une synthèse courte et actionnable.\n\n"
            f"CONTENU:\n{scraped_text[:15000]}"
        )
        return call_gemini_with_retry(prompt=fallback_prompt) or "Aucune description disponible pour cette URL."
    return "Aucune description disponible pour cette URL."


# ---------------------------------------------------------------------------
# Search helper (optional)

def perform_search(
    keyword: str, site: str, max_results: int = 5, time_window_days: int = 1
) -> List[Dict[str, str]]:
    """
    Perform a search for the given keyword on a specific site using an external
    search API (e.g. Google Custom Search or Bing). Returns a list of results
    with keys: title, url, snippet. If no API is configured, returns an empty list.

    You can configure the search engine by setting environment variables:
      - GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX for Google Custom Search Engine
      - BING_API_KEY for Bing Web Search
    The time_window_days parameter may be ignored depending on the API used.
    """
    results: List[Dict[str, str]] = []
    # Google Custom Search
    g_key = os.getenv("GOOGLE_CSE_API_KEY")
    g_cx = os.getenv("GOOGLE_CSE_CX")
    if g_key and g_cx:
        try:
            query = f"site:{site} {keyword}"
            # restrict to last X days is not directly supported; we rely on API ranking
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": g_key,
                "cx": g_cx,
                "q": query,
                "num": max_results,
            }
            resp = http_client.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                for item in items:
                    link = item.get("link")
                    title = item.get("title")
                    snippet = item.get("snippet")
                    if link:
                        results.append(
                            {
                                "title": title or "",
                                "url": link,
                                "snippet": snippet or "",
                            }
                        )
                return results
            else:
                logger.warning(
                    f"Google CSE API returned status {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            logger.warning(f"Google CSE API error: {e}")

    # Bing Search (via RapidAPI or similar)
    bing_key = os.getenv("BING_API_KEY")
    if bing_key:
        try:
            query = f"site:{site} {keyword}"
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": bing_key}
            params = {"q": query, "count": max_results}
            resp = http_client.session.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                web_pages = data.get("webPages", {}).get("value", [])
                for item in web_pages:
                    link = item.get("url")
                    title = item.get("name")
                    snippet = item.get("snippet")
                    if link:
                        results.append(
                            {
                                "title": title or "",
                                "url": link,
                                "snippet": snippet or "",
                            }
                        )
                return results
            else:
                logger.warning(
                    f"Bing API returned status {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            logger.warning(f"Bing API error: {e}")

    # No API configured or all failed
    return results


# ---------------------------------------------------------------------------
# Pydantic models

class SourceConfig(BaseModel):
    keywords: List[str] = []
    veille_par_url: List[str] = []


class UpdateSourcesRequest(BaseModel):
    add_subjects: Optional[List[str]] = None
    add_urls: Optional[List[str]] = None
    remove_subjects: Optional[List[str]] = None
    remove_urls: Optional[List[str]] = None
    replace: Optional[SourceConfig] = None


class UpdateRecipientsRequest(BaseModel):
    add: Optional[List[str]] = None
    remove: Optional[List[str]] = None
    replace: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Core watch logic with concurrency control

async def perform_watch_task() -> None:
    """
    Main watch task. Processes the configured keywords and URLs, fetches and
    summarizes new content, updates the memory, and sends email reports.

    Uses a simple file lock to prevent concurrent execution.
    """
    # Acquire lock
    try:
        # Attempt to create lock file exclusively
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(time.time()))
    except FileExistsError:
        # Another watch is running
        logger.info("Watch task is already running. Skipping.")
        return
    except Exception as e:
        logger.error(f"Failed to acquire watch lock: {e}")
        return

    try:
        logger.info("Watch task started.")
        # Load configuration
        try:
            config_data = safe_load_json(SOURCES_FILE, {})
            config = SourceConfig(**config_data)
        except (ValidationError, Exception) as e:
            logger.error(f"Invalid sources configuration: {e}. Aborting watch.")
            return
        keywords = config.keywords or []
        urls_to_watch = config.veille_par_url or []
        # Normalize URLs to watch
        urls_to_watch = [normalize_url(u) for u in urls_to_watch if u]
        # Load memory
        memory = safe_load_memory()
        seen_urls_set: Set[str] = set(memory.get("seen_urls", []))
        new_urls: Set[str] = set()
        new_details: Dict[str, Any] = {}
        findings: List[Dict[str, Any]] = []
        # Process keywords × URLs
        for keyword in keywords:
            if not keyword:
                continue
            for base_url in urls_to_watch:
                # # Perform external search for latest results
                # search_results = perform_search(keyword, base_url)
                # for res in search_results:
                #     link = normalize_url(res["url"])
                #     if not link or link in seen_urls_set:
                #         continue
                #     # Fetch page text
                #     title, text = fetch_url_text(link)
                #     if not text:
                #         continue
                #     # Check if keyword appears in text (basic filter)
                #     if keyword.lower() not in text.lower():
                #         continue
                #     # Summarize
                #     summary = summarize_with_url_context(link, text)
                #     if not summary:
                #         continue
                #     findings.append(
                #         {
                #             "keyword": keyword,
                #             "url": link,
                #             "title": title or res.get("title") or "Sans titre",
                #             "summary": summary,
                #             "source": "search",
                #             "signal": "search",
                #         }
                #     )
                # # Directly analyse base URL if not already processed
                if base_url and base_url not in seen_urls_set:
                    title, text = fetch_url_text(base_url)
                    if text and keyword.lower() in text.lower():
                        summary = summarize_with_url_context(base_url, text)
                        findings.append(
                            {
                                "keyword": keyword,
                                "url": base_url,
                                "title": title or "Sans titre",
                                "summary": summary,
                                "source": "direct_url",
                                "signal": "direct_url",
                            }
                        )
        # Filter out duplicates and seen URLs
        unique_findings: List[Dict[str, Any]] = []
        for item in findings:
            url = item["url"]
            if url not in seen_urls_set:
                unique_findings.append(item)
                new_urls.add(url)
                new_details[url] = {
                    "title": item["title"],
                    "summary": item["summary"],
                    "matched_keywords": [item["keyword"]],
                    "source": item["source"],
                    "signal": item["signal"],
                }
                seen_urls_set.add(url)
        # Build report
        report_text = ""
        if unique_findings:
            # Group by keyword
            keyword_to_items: Dict[str, List[Dict[str, Any]]] = {}
            for item in unique_findings:
                kw = item["keyword"]
                keyword_to_items.setdefault(kw, []).append(item)
            lines = ["# Rapport de veille par thématique\n"]
            for kw, items in keyword_to_items.items():
                lines.append(f"\n## Thématique: {kw}\n")
                for it in items:
                    t = it.get("title") or "Sans titre"
                    u = it.get("url")
                    s = it.get("summary") or "Pas de résumé disponible"
                    src = it.get("source")
                    sig = it.get("signal")
                    lines.append(
                        f"- {t}\n  {u}\n  Source: {src}\n  Signal: {sig}\n  {s}\n"
                    )
            report_prompt = (
                "Rédige un rapport synthétique (en français) sur les actualités suivantes, "
                "organisé par thématique. Pour chaque thématique, résume les points clés "
                "et termine par une synthèse globale avec 2-3 recommandations actionnables.\n\n"
                + "\n".join(lines)
            )
            generated_report = call_gemini_with_retry(prompt=report_prompt)
            report_text = generated_report or "\n".join(lines)
        # Persist memory and send report
        if new_urls:
            memory["seen_urls"] = list(seen_urls_set)
            memory_details = memory.get("details", {})
            memory_details.update(new_details)
            memory["details"] = memory_details
            if report_text:
                report_entry = {
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "new_urls": sorted(new_urls),
                    "report": report_text,
                }
                memory_reports = memory.get("reports", [])
                memory_reports.append(report_entry)
                memory["reports"] = memory_reports
                # Send email
                send_report_via_email(
                    subject=f"Rapport de veille - {len(new_urls)} nouvelles actualités",
                    body=report_text,
                )
            # Save memory
            atomic_save_memory(memory)
        else:
            logger.info("No new relevant content found.")
        logger.info("Watch task completed.")
    finally:
        # Release lock
        try:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Email sending

def send_report_via_email(subject: str, body: str) -> None:
    """
    Send the report via email to all configured recipients using an external
    mail API. If no recipients are configured or requests is unavailable, the
    send is skipped.
    """
    recipients = safe_load_recipients()
    if not recipients:
        logger.info("No recipients configured. Skipping email.")
        return
    # Compose payload
    to_email = ", ".join(recipients)
    payload = {
        "to": to_email,
        "cc": "",
        "bcc": "",
        "subject": subject,
        "message": body,
        "isHtml": False,
        "attachments": [],
    }
    try:
        response = http_client.session.post(
            "https://mail-api-mounsef.vercel.app/api/send-email",
            json=payload,
            timeout=15,
        )
        if response.ok:
            logger.info(f"Report email successfully sent to {len(recipients)} recipients.")
        else:
            try:
                res_json = response.json()
                err = res_json.get("error", "Unknown error")
            except Exception:
                err = response.text
            logger.error(f"Failed to send email: {err}")
    except Exception as e:
        logger.error(f"Error sending email: {e}")


# ---------------------------------------------------------------------------
# FastAPI app

app = FastAPI(
    title="Watcher API v3",
    description=(
        "API de veille stratégique améliorée avec scraping, recherche externe facultative "
        "et contexte URL pour Gemini. La mémoire est bornée et la concurrence contrôlée."
    ),
)

# Enable open CORS as requested. In production you should restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/watch", summary="Déclenche la veille en tâche de fond")
async def trigger_watch_endpoint(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Launch the watch task asynchronously. If a watch is already running, it will
    skip launching another instance.
    """
    background_tasks.add_task(perform_watch_task)
    logger.info("Watch request received. Task scheduled in background.")
    return {"message": "La veille a été lancée en arrière-plan. Consultez les logs pour les détails."}


@app.get("/memory", summary="Affiche la mémoire complète")
async def get_memory_content() -> Dict[str, Any]:
    """
    Return the entire memory content. Use with caution as this may include a large
    number of entries.
    """
    return safe_load_memory()


@app.get("/", summary="Endpoint de santé")
async def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Watcher API is operational."}


@app.get("/sources", summary="Lire la configuration des sources")
async def read_sources() -> Dict[str, Any]:
    """
    Read the current sources configuration.
    """
    data = safe_load_json(SOURCES_FILE, {})
    try:
        config = SourceConfig(**data)
        return config.dict()
    except (ValidationError, Exception) as e:
        raise HTTPException(status_code=500, detail=f"Invalid sources configuration: {e}")


@app.post("/sources", summary="Modifier la configuration des sources")
async def update_sources(update: UpdateSourcesRequest) -> Dict[str, Any]:
    """
    Update the sources configuration. Supports adding/removing keywords and URLs,
    or replacing the entire configuration.
    """
    current_data = safe_load_json(SOURCES_FILE, {})
    try:
        current_config = SourceConfig(**current_data)
    except (ValidationError, Exception):
        current_config = SourceConfig()
    if update.replace is not None:
        new_config = update.replace
    else:
        new_config = SourceConfig(
            keywords=list(current_config.keywords),
            veille_par_url=list(current_config.veille_par_url),
        )
        if update.add_subjects:
            for subj in update.add_subjects:
                if subj and subj not in new_config.keywords:
                    new_config.keywords.append(subj)
        if update.add_urls:
            for url in update.add_urls:
                nurl = normalize_url(url)
                if nurl and nurl not in new_config.veille_par_url:
                    new_config.veille_par_url.append(nurl)
        if update.remove_subjects:
            new_config.keywords = [s for s in new_config.keywords if s not in (update.remove_subjects or [])]
        if update.remove_urls:
            to_remove = {normalize_url(u) for u in update.remove_urls}
            new_config.veille_par_url = [u for u in new_config.veille_par_url if u not in to_remove]
    # Save
    try:
        atomic_save_json(new_config.dict(), SOURCES_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing sources: {e}")
    return new_config.dict()


@app.get("/details", summary="Consulter les descriptions enregistrées")
async def get_details() -> Dict[str, Any]:
    """
    Return the stored details for all seen URLs.
    """
    memory = safe_load_memory()
    return memory.get("details", {})


@app.get("/reports", summary="Consulter l'historique des rapports générés")
async def get_reports() -> List[Dict[str, Any]]:
    """
    Return the list of generated reports.
    """
    memory = safe_load_memory()
    return memory.get("reports", [])


@app.get("/recipients", summary="Obtenir la liste des destinataires")
async def get_recipients() -> List[str]:
    """
    Return the current list of email recipients.
    """
    return safe_load_recipients()


@app.post("/recipients", summary="Modifier la liste des destinataires")
async def update_recipients(update: UpdateRecipientsRequest) -> List[str]:
    """
    Update the list of email recipients. Supports adding, removing, or replacing.
    """
    current = safe_load_recipients()
    if update.replace is not None:
        new_list = [str(addr).strip() for addr in (update.replace or []) if addr]
        # Validate emails
        new_valid: List[str] = []
        for addr in new_list:
            if re.match(r"^[^@]+@[^@]+\.[^@]+$", addr):
                new_valid.append(addr)
            else:
                logger.warning(f"Ignoring invalid email address: {addr}")
        new_list = new_valid
    else:
        new_list = list(current)
        if update.add:
            for addr in update.add:
                if addr and addr not in new_list and re.match(r"^[^@]+@[^@]+\.[^@]+$", addr):
                    new_list.append(addr.strip())
        if update.remove:
            new_list = [addr for addr in new_list if addr not in (update.remove or [])]
    atomic_save_recipients(new_list)
    return sorted(set(new_list))
