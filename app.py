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

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and configuration

# Directory for persisting configuration and state
DATA_DIR = os.getenv("WATCHER_DATA_DIR", ".")

# Files for persisting sources, memory and recipients
SOURCES_FILE: str = os.path.join(DATA_DIR, "sources.json")
MEMORY_FILE: str = os.path.join(DATA_DIR, "memory_db.json")
RECIPIENTS_FILE: str = os.path.join(DATA_DIR, "recipients.json")

# Default in‑memory structure
DEFAULT_MEMORY: Dict[str, Any] = {
    "seen_urls": [],
    "details": {},
    "reports": [],
}

# Limits for memory – keeps disk usage bounded
MAX_SEEN_URLS = int(os.getenv("WATCHER_MAX_SEEN_URLS", "10000"))
MAX_REPORTS = int(os.getenv("WATCHER_MAX_REPORTS", "100"))

# User agent for HTTP requests
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; WatcherBot/3.0; +https://example.com/bot) "
    "PythonRequests"
)

# Lock file to avoid concurrent watcher runs
LOCK_FILE = os.path.join(DATA_DIR, ".watch_lock")

# ---------------------------------------------------------------------------
# OpenAI configuration
#
# We use OpenAI's Chat Completions with the web search tool enabled to
# summarise pages and compile reports. The model name and the default
# location used for search results can be overridden via environment variables.
#
# See OpenAI documentation for details on the `search_context_size` parameter,
# which controls how much context is retrieved from the web【122901823563527†L448-L472】,
# and the `user_location` parameter, which biases search results to a region【122901823563527†L368-L374】.
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set. API calls will fail.")

OPENAI_SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-search-preview")
OPENAI_LOCATION = {
    "country": os.getenv("WATCHER_COUNTRY", "MA"),
    "city": os.getenv("WATCHER_CITY", "Casablanca"),
    "region": os.getenv("WATCHER_REGION", "Casablanca-Settat"),
}
oa_client = OpenAI(api_key=OPENAI_API_KEY)

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
# OpenAI helpers

def call_openai_with_search(
    prompt: str,
    max_retries: int = 2,
    initial_delay: int = 5,
    model_name: str = OPENAI_SEARCH_MODEL,
    search_context_size: str = "medium",
) -> Dict[str, Any]:
    """
    Call OpenAI's Chat Completions API with the web search tool enabled.

    Returns a dictionary with keys:
      - 'text': the generated text (may include inline citations)
      - 'citations': list of citations (URL, title, start, end indices)

    Retries on errors up to `max_retries` times with exponential backoff.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = oa_client.chat.completions.create(
                model=model_name,
                web_search_options={
                    "search_context_size": search_context_size,
                    "user_location": {
                        "type": "approximate",
                        "approximate": OPENAI_LOCATION,
                    },
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a monitoring assistant. "
                            "Answer concisely in the language of the prompt and include inline citations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            citations: List[Dict[str, Any]] = []
            for ann in getattr(msg, "annotations", []) or []:
                if getattr(ann, "type", "") == "url_citation" and hasattr(ann, "url_citation"):
                    uc = ann.url_citation
                    citations.append(
                        {
                            "url": uc.url,
                            "title": uc.title,
                            "start": uc.start_index,
                            "end": uc.end_index,
                        }
                    )
            return {"text": content.strip(), "citations": citations}
        except Exception as e:
            last_error = e
            logger.error(f"OpenAI web search error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # exponential backoff
                time.sleep(initial_delay * (2 ** attempt))
    logger.error(f"All OpenAI calls failed. Last error: {last_error}")
    return {"text": "", "citations": []}


def summarize_with_url_context(url: str, scraped_text: str) -> str:
    """
    Summarise the content of a URL using OpenAI's web search tool. The URL
    itself is provided in the prompt to anchor the search. If the call
    fails or returns empty, a fallback summarisation using the scraped text
    is attempted.
    """
    # Primary prompt referencing the URL
    url_prompt = (
        "Analyse et résume précisément en français le contenu de cette URL. "
        "Mets en avant les nouvelles informations et points clés, en indiquant les dates si elles sont présentes. "
        "Structure la réponse en puces, suivie d'un court paragraphe de synthèse. "
        "Ajoute des citations en ligne pour les sources utilisées."
    )
    try:
        combined_prompt = f"{url_prompt}\n\nURL: {url}"
        result = call_openai_with_search(prompt=combined_prompt, search_context_size="medium")
        if result["text"]:
            return result["text"]
    except Exception as e:
        logger.info(f"OpenAI URL summarisation failed for {url}: {e}")
    # Fallback: summarise the scraped text directly
    if scraped_text:
        fallback_prompt = (
            "Voici le contenu d'une page web. Résume-le en français, en listant d'abord les points clés, "
            "puis une synthèse courte et actionnable. Ajoute des citations si possible.\n\n"
            f"CONTENU:\n{scraped_text[:15000]}"
        )
        result = call_openai_with_search(prompt=fallback_prompt, search_context_size="low")
        if result["text"]:
            return result["text"]
    return "Aucune description disponible pour cette URL."


def perform_search_openai(
    keyword: str,
    site: str,
    max_results: int = 5,
    time_window_hours: int = 48,
) -> List[Dict[str, str]]:
    """
    Use OpenAI's web search tool to find recent pages for `site` and `keyword`.
    Returns a list of dictionaries with keys: title, url, snippet.
    If parsing fails or the model returns non-JSON, an empty list is returned.
    """
    prompt = (
        "Find the most recent pages in the last {hrs} hours relevant to the following search query:\n"
        f"site:{site} {keyword}\n\n"
        "Return a STRICT JSON array of objects with keys: title, url, snippet. "
        "Do not include any extra text before or after the JSON. Limit the list to {limit} items."
    ).format(hrs=time_window_hours, limit=max_results)
    result = call_openai_with_search(prompt=prompt, search_context_size="low")
    text = result.get("text", "")
    if not text:
        return []
    # Try to parse JSON from the model output
    try:
        data = json.loads(text)
        if isinstance(data, list):
            parsed_items: List[Dict[str, str]] = []
            for item in data[:max_results]:
                try:
                    parsed_items.append(
                        {
                            "title": str(item.get("title", "")),
                            "url": str(item.get("url", "")),
                            "snippet": str(item.get("snippet", "")),
                        }
                    )
                except Exception:
                    continue
            return parsed_items
    except Exception:
        pass
    # Fallback: attempt to extract JSON array from text
    try:
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.S)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                parsed_items: List[Dict[str, str]] = []
                for item in data[:max_results]:
                    try:
                        parsed_items.append(
                            {
                                "title": str(item.get("title", "")),
                                "url": str(item.get("url", "")),
                                "snippet": str(item.get("snippet", "")),
                            }
                        )
                    except Exception:
                        continue
                return parsed_items
    except Exception:
        pass
    return []


# New helper: watch a site for multiple keywords using OpenAI web search.
def watch_site_for_keywords(site: str, keywords: List[str]) -> List[Dict[str, str]]:
    """
    Use OpenAI's web search to perform a watch task on an entire website for a list of keywords.

    For the given `site` (e.g. 'https://example.com'), this function instructs OpenAI to search
    for all pages or articles published in the last 48 hours that relate to any of the provided
    keywords. The assistant is expected to return a JSON array of objects where each object has
    the following keys:

      - "Source": the name or title of the publication or source
      - "Contexte et Résumé de la publication": a concise French summary of the content
      - "Date de Publication": the publication date of the content (ISO or human readable)
      - "Implications et Impacts sur UM6P": analysis of how the content affects UM6P
      - "Recommandations Stratégiques pour UM6P": actionable recommendations for UM6P
      - "Lien": the URL of the original content

    Returns an empty list if no results or if parsing fails.
    """
    if not keywords:
        return []
    keywords_str = ", ".join(keywords)
    # Construct prompt instructing the model to perform a comprehensive site search.
    prompt = (
        "Vous êtes un assistant de veille stratégique. "
        "Sur le site suivant: {site}, recherchez toutes les publications des dernières 48 heures "
        "qui traitent des mots-clés suivants: {keywords}. "
        "Pour chaque publication trouvée, renvoyez un tableau JSON (array) d'objets avec les champs suivants: "
        "\"Source\", \"Contexte et Résumé de la publication\", \"Date de Publication\", "
        "\"Implications et Impacts sur UM6P\", \"Recommandations Stratégiques pour UM6P\", \"Lien\". "
        "Incluez uniquement les publications publiées au cours des dernières 48 heures. "
        "Le résultat doit être STRICTEMENT du JSON sans aucun texte supplémentaire avant ou après."
    ).format(site=site, keywords=keywords_str)
    result = call_openai_with_search(prompt=prompt, search_context_size="high")
    text = result.get("text", "")
    if not text:
        return []
    # Attempt to parse JSON array from the model output
    try:
        data = json.loads(text)
        if isinstance(data, list):
            parsed_items: List[Dict[str, str]] = []
            for item in data:
                if isinstance(item, dict):
                    parsed_items.append(item)
            return parsed_items
    except Exception:
        pass
    # Fallback: extract the first JSON array in the output if extra text is present
    try:
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                parsed_items: List[Dict[str, str]] = []
                for item in data:
                    if isinstance(item, dict):
                        parsed_items.append(item)
                return parsed_items
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Pydantic models for request validation

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
    summarises new content, updates the memory, and sends email reports.

    A file lock is used to prevent concurrent executions.
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
        all_results: List[Dict[str, Any]] = []
        # Watch each site for the list of keywords using OpenAI
        for site in urls_to_watch:
            if not site:
                continue
            try:
                site_results = watch_site_for_keywords(site, keywords)
            except Exception as e:
                logger.error(f"Error watching site {site}: {e}")
                site_results = []
            for entry in site_results:
                # Extract and normalize the URL of the publication
                link = normalize_url(str(entry.get("Lien", "")))
                if not link:
                    continue
                if link in seen_urls_set:
                    continue
                # Mark as seen and accumulate
                seen_urls_set.add(link)
                new_urls.add(link)
                new_details[link] = entry
                all_results.append(entry)
        # Build report
        report_text = ""
        if all_results:
            # Compose prompt asking the model to create a concise Markdown table and summary
            try:
                json_results = json.dumps(all_results, ensure_ascii=False)
            except Exception:
                json_results = str(all_results)
            report_prompt = (
                "Vous êtes un assistant de veille stratégique. "
                "À partir de la liste JSON suivante d'actualités, génère un rapport structuré "
                "sous forme de tableau Markdown. Le tableau doit avoir les colonnes suivantes : "
                "Source, Contexte et Résumé de la publication, Date de Publication, "
                "Implications et Impacts sur UM6P, Recommandations Stratégiques pour UM6P, Lien. "
                "Chaque ligne du tableau doit synthétiser l'entrée correspondante avec des phrases courtes "
                "et éviter les longues descriptions. "
                "Après le tableau, ajoute un court paragraphe de synthèse générale (2-3 phrases) "
                "et 2-3 recommandations actionnables pour UM6P. "
                "Voici la liste JSON:\n\n"
                f"{json_results}"
            )
            result = call_openai_with_search(prompt=report_prompt, search_context_size="high")
            report_text = result.get("text", "")
            if not report_text:
                # Build fallback table manually
                headers = [
                    "Source",
                    "Contexte et Résumé de la publication",
                    "Date de Publication",
                    "Implications et Impacts sur UM6P",
                    "Recommandations Stratégiques pour UM6P",
                    "Lien",
                ]
                lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
                for entry in all_results:
                    row = [
                        str(entry.get("Source", "")) or "",
                        str(entry.get("Contexte et Résumé de la publication", "")) or "",
                        str(entry.get("Date de Publication", "")) or "",
                        str(entry.get("Implications et Impacts sur UM6P", "")) or "",
                        str(entry.get("Recommandations Stratégiques pour UM6P", "")) or "",
                        str(entry.get("Lien", "")) or "",
                    ]
                    lines.append(" | ".join(row))
                report_text = "\n".join(lines)
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
        "API de veille stratégique avec scraping, recherche web via OpenAI et contexte URL. "
        "La mémoire est bornée et la concurrence contrôlée."
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
