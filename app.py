import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Set, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import Tool
from google.generativeai.types.generation_types import GenerationConfig
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

import smtplib
from email.message import EmailMessage

# fix cors problems
from fastapi.middleware.cors import CORSMiddleware

# NEW: lightweight scraping
try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# -----------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Gemini API configuration
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.critical(
        "Erreur critique: la variable d'environnement GEMINI_API_KEY n'est pas définie."
    )
    raise SystemExit(
        "GEMINI_API_KEY non définie. Veuillez définir cette variable d'environnement et redémarrer."
    )

# Default Gemini model
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# -----------------------------------------------------------------------------
# File paths and defaults
SOURCES_FILE: str = os.getenv("SOURCES_FILE", "sources.json")
MEMORY_FILE: str = os.getenv("MEMORY_FILE", "memory_db.json")

DEFAULT_MEMORY: Dict[str, Any] = {
    "seen_urls": [],
    "details": {},  # url -> {"title": str, "summary": str, "text": str}
    "reports": [],
}

# Additional configuration files for recipients. The existing sources file
# (SOURCES_FILE) already stores the list of keywords (veille_par_sujet) and
# URLs (veille_par_url). We introduce a file to manage email recipients.
RECIPIENTS_FILE: str = os.getenv("RECIPIENTS_FILE", "recipients.json")

# -----------------------------------------------------------------------------
# Email configuration


def safe_load_recipients(path: str = RECIPIENTS_FILE) -> List[str]:
    """
    Charge la liste des destinataires depuis un fichier JSON. Retourne une
    liste vide en cas d'erreur ou si le fichier n'existe pas.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning(f"Le fichier '{path}' ne contient pas une liste. Reset destinataires.")
            return []
        return [str(item).strip() for item in data if item]
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Lecture des destinataires '{path}' impossible: {e}. Reset.")
        return []


def atomic_save_recipients(recipients: List[str], path: str = RECIPIENTS_FILE) -> None:
    """
    Enregistre la liste des destinataires de façon atomique.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".recipients_tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(sorted(set(recipients)), tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def send_report_via_email(subject: str, body: str) -> None:
    """
    Envoie le rapport par email à tous les destinataires configurés en utilisant
    l'API https://mail-api-mounsef.vercel.app/api/send-email.
    Si aucun destinataire n'est configuré, l'envoi est ignoré avec un message de log.
    """
    if requests is None:
        logger.error("Le module 'requests' n'est pas installé. Impossible d'envoyer l'email.")
        return
        
    recipients = safe_load_recipients()
    if not recipients:
        logger.info("Aucun destinataire configuré, aucun email envoyé.")
        return
    
    try:
        to_email = ", ".join(recipients)
        
        payload = {
            "to": to_email,
            "cc": "",
            "bcc": "",
            "subject": subject,
            "message": body,
            "isHtml": False,
            "attachments": []
        }
        
        response = requests.post(
            'https://mail-api-mounsef.vercel.app/api/send-email',
            headers={'Content-Type': 'application/json'},
            json=payload
        )
        
        if response.ok:
            logger.info(f"Rapport envoyé avec succès à {len(recipients)} destinataires.")
        else:
            result = response.json()
            logger.error(f"Échec de l'envoi de l'email: {result.get('error', 'Erreur inconnue')}")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email: {e}")

# -----------------------------------------------------------------------------
# Pydantic models
class SourceConfig(BaseModel):
    veille_par_sujet: List[str] = []
    veille_par_url: List[str] = []


class UpdateSourcesRequest(BaseModel):
    add_subjects: Optional[List[str]] = None
    add_urls: Optional[List[str]] = None
    remove_subjects: Optional[List[str]] = None
    remove_urls: Optional[List[str]] = None
    replace: Optional[SourceConfig] = None


# Recipients management models
class UpdateRecipientsRequest(BaseModel):
    add: Optional[List[str]] = None
    remove: Optional[List[str]] = None
    replace: Optional[List[str]] = None


# -----------------------------------------------------------------------------
# Memory helpers
def safe_load_memory(path: str = MEMORY_FILE) -> Dict[str, Any]:
    """
    Charge le contenu du fichier mémoire en toute sécurité. Si le fichier est
    inexistant ou mal formé, retourne une mémoire vierge.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "seen_urls" not in data:
            logger.warning(
                f"Format de fichier mémoire invalide pour '{path}'. Réinitialisation de la mémoire."
            )
            return DEFAULT_MEMORY.copy()
        memory: Dict[str, Any] = {
            "seen_urls": data.get("seen_urls", []),
            "details": data.get("details", {}),
            "reports": data.get("reports", []),
        }
        # Validate types
        if not isinstance(memory["seen_urls"], list):
            logger.warning(f"'seen_urls' n'est pas une liste. Reset.")
            return DEFAULT_MEMORY.copy()
        if not isinstance(memory["details"], dict):
            logger.warning(f"'details' n'est pas un objet. Reset.")
            return DEFAULT_MEMORY.copy()
        if not isinstance(memory["reports"], list):
            logger.warning(f"'reports' n'est pas une liste. Reset.")
            return DEFAULT_MEMORY.copy()
        return memory
    except FileNotFoundError:
        logger.info(f"Fichier mémoire '{path}' non trouvé. Initialisation.")
        return DEFAULT_MEMORY.copy()
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Lecture mémoire '{path}' impossible: {e}. Reset.")
        return DEFAULT_MEMORY.copy()


def atomic_save_memory(memory: Dict[str, Any], path: str = MEMORY_FILE) -> None:
    """
    Enregistre la mémoire de manière atomique pour éviter la corruption de fichier.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    mem_copy = memory.copy()
    mem_copy["seen_urls"] = sorted(set(mem_copy.get("seen_urls", [])))
    fd, temp_path = tempfile.mkstemp(prefix=".memory_tmp_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(mem_copy, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, path)
        logger.info(
            f"Sauvegarde mémoire OK. URLs: {len(mem_copy['seen_urls'])}."
        )
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Scraping helpers (NEW)
USER_AGENT = (
    "Mozilla/5.0 (compatible; WatcherBot/1.0; +https://example.com/bot) "
    "PythonRequests"
)


def _clean_text(text: str) -> str:
    """
    Normalise les espaces et supprime les lignes vides en excès.
    """
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_main_text(html: str) -> Tuple[str, str]:
    """
    Extrait le titre et le texte principal d'un document HTML. Utilise BeautifulSoup
    si disponible, sinon effectue un décapage basique des balises.
    """
    if not html:
        return ("", "")
    if BeautifulSoup is None:
        # fallback: strip tags crudely
        title_match = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        title = title_match.group(1).strip() if title_match else ""
        # remove tags
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return (title, _clean_text(text))

    soup = BeautifulSoup(html, "html.parser")

    # remove scripts/styles and common non-content containers
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ["nav", "footer", "header", "form", "aside"]:
        for t in soup.select(sel):
            t.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()

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
    Tente de récupérer le contenu HTML d'une URL et d'en extraire le titre et le
    texte principal. En cas d'échec, retourne des chaînes vides.
    """
    if requests is None:
        logger.warning("Le module 'requests' n'est pas installé. Impossible de scraper.")
        return ("", "")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
        )
        if resp.status_code >= 400:
            logger.warning(f"Échec HTTP {resp.status_code} pour {url}")
            return ("", "")
        html = resp.text
        return extract_main_text(html)
    except Exception as e:
        logger.warning(f"Erreur réseau pour {url}: {e}")
        return ("", "")


# -----------------------------------------------------------------------------
# Gemini helpers

def call_gemini_with_retry(
    prompt: str,
    max_retries: int = 3,
    initial_delay: int = 5,
    model_name: str = "gemini-2.5-flash"
) -> str:
    """Call the Gemini API with retry logic and return the response text.

    This helper abstracts away repeated attempts to contact the model.  It does
    not raise on failure; instead, it logs errors and returns an empty string
    after exhausting retries.

    Args:
        prompt: The textual prompt to send to the model.
        max_retries: Maximum number of attempts before giving up.
        initial_delay: Initial backoff delay (seconds) between retries.
        model_name: Name of the Gemini model to use.
    Returns:
        The model's textual response, or an empty string if all attempts fail.
    """

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
                        generation_config=GenerationConfig(
                            tools=tools,
                        )
                    )

            # The API returns an object where `.text` holds the plain content.
            # Fallback to candidate text if `.text` is missing.
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            # Some SDK versions nest the text inside `candidates[0]`.
            if getattr(response, "candidates", None):
                candidate = response.candidates[0]
                text = getattr(candidate, "text", "") or getattr(candidate, "content", "")
                if text:
                    return str(text).strip()
            logger.warning(
                f"Réponse vide de Gemini (tentative {attempt + 1}/{max_retries}) pour le prompt: {prompt}"
            )
        except Exception as e:
            logger.error(
                f"Erreur lors de l'appel à Gemini (tentative {attempt + 1}/{max_retries}) pour le prompt '{prompt}': {e}"
            )
        # If there are remaining attempts, sleep before the next try.
        if attempt < max_retries - 1:
            sleep_time = initial_delay * (2 ** attempt)
            logger.info(f"Nouvelle tentative dans {sleep_time} secondes...")
            time.sleep(sleep_time)
    logger.error(
        f"Toutes les tentatives ont échoué pour le prompt '{prompt}'. Retour d'une chaîne vide."
    )
    return ""


def summarize_with_url_context(url: str, scraped_text: str) -> str:
    """
    Tente d'analyser et de résumer le contenu d'une URL. Si l'utilisation du
    contexte URL n'est pas disponible, s'appuie sur le texte scrappé. Cette
    fonction n'utilise pas de 'url' part direct, car le SDK ne reconnaît pas un
    dictionnaire avec la clé 'url' seule【649065952783530†L207-L260】. À la place,
    on fournit l'URL comme simple texte dans le prompt. En cas d'échec ou si
    aucun texte n'est disponible, retourne un message par défaut.
    """
    # 1) Prompt pour demander un résumé en mentionnant l'URL. On évite d'utiliser
    # un 'url' part non pris en charge par le SDK.
    url_prompt = (
        "Analyse et résume précisément en français le contenu de cette URL. "
        "Mets en avant les mises à jour, nouvelles informations et points clés. "
        "Structure la réponse avec des puces claires et un court paragraphe de synthèse à la fin."
    )

    # On construit un prompt textuel qui inclut explicitement l'URL. Le modèle
    # pourra utiliser ses connaissances ou échouer silencieusement si la
    # récupération n'est pas possible.
    try:
        combined_prompt = f"{url_prompt}\n\nURL: {url}"
        text = call_gemini_with_retry(prompt=combined_prompt)
        if text:
            return text
    except Exception as e:
        logger.info(f"Contexte URL non disponible ou a échoué pour {url}: {e}")

    # 2) Fallback: si nous disposons du texte scrappé, résume-le.
    if scraped_text:
        fallback_prompt = (
            "Voici le contenu d'une page web. Résume-le en français, en listant d'abord les points clés, "
            "puis une synthèse courte et actionnable.\n\n"
            f"CONTENU:\n{scraped_text[:15000]}"
        )
        return call_gemini_with_retry(prompt=fallback_prompt) or "Aucune description disponible pour cette URL."
    # 3) Dernier recours
    return "Aucune description disponible pour cette URL."


# -----------------------------------------------------------------------------
# Veille logic
async def perform_watch_task() -> None:
    logger.info("Tâche de veille démarrée.")
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = SourceConfig(**config_data)
    except FileNotFoundError:
        logger.error(f"Fichier de configuration '{SOURCES_FILE}' introuvable. Veille annulée.")
        return
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Config '{SOURCES_FILE}' invalide: {e}. Veille annulée.")
        return

    subjects_to_watch = config.veille_par_sujet
    urls_to_watch = config.veille_par_url

    memory = safe_load_memory()
    seen_urls_set: Set[str] = set(memory.get("seen_urls", []))
    details: Dict[str, Any] = memory.get("details", {})
    new_urls: Set[str] = set()
    new_details: Dict[str, Any] = {}

    # Step 1: Start with all URLs from configuration
    urls_to_process = list(urls_to_watch)
    logger.info(f"Traitement de {len(urls_to_process)} URLs depuis la configuration")
    
    # Step 2: Scrape each URL and check for keywords
    keyword_matches = {}  # URL -> [matched_keywords]
    
    # Préparer la liste des mots-clés en minuscules pour la recherche dans les pages
    keywords_lower = [kw.lower() for kw in subjects_to_watch if kw]
    
    # Boucle de traitement des URLs
    for url in urls_to_process:
        if url in seen_urls_set:
            logger.info(f"URL déjà analysée précédemment: '{url}'")
            continue
            
        title, text = fetch_url_text(url)
        if not text:
            logger.info(f"Impossible de récupérer le contenu pour '{url}'")
            continue
        
        # Détermine si la page contient au moins un des mots-clés suivis
        text_lower = text.lower()
        matched_keywords = [kw for kw in keywords_lower if kw in text_lower]
        
        if not matched_keywords:
            logger.info(f"Aucun mot-clé trouvé dans '{url}', page ignorée")
            continue
            
        # Si on arrive ici, on a trouvé au moins un mot-clé
        keyword_matches[url] = matched_keywords
        logger.info(f"URL '{url}' contient les mots-clés: {', '.join(matched_keywords)}")
        new_urls.add(url)

    # Step 3: For each URL with keywords, generate summaries focused on those keywords
    for url, matched_keywords in keyword_matches.items():
        title, text = fetch_url_text(url)  # Re-fetch or use cached content
        
        # Créer un prompt qui met l'accent sur les mots-clés trouvés
        context_prompt = (
            f"Analyse cette page web et résume les actualités ou informations liées spécifiquement "
            f"aux thématiques suivantes: {', '.join(matched_keywords)}.\n\n"
            f"CONTENU DE LA PAGE:\n{text[:15000]}"
        )
        
        summary = call_gemini_with_retry(prompt=context_prompt)
        if not summary:
            summary = f"Mots-clés trouvés: {', '.join(matched_keywords)}. Impossible de générer un résumé détaillé."
        
        new_details[url] = {
            "title": title or "",
            "summary": summary,
            "matched_keywords": matched_keywords,
            "text": text[:5000] if text else "",  # Limiter la taille pour économiser l'espace
        }
        logger.info(f"Résumé généré pour '{url}' avec les mots-clés {matched_keywords}")

    # Step 4: Build a synthetic report of all new items, organized by keyword
    report_text = ""
    if new_details:
        # Regrouper par mot-clé
        keyword_to_urls = {}
        for url, info in new_details.items():
            for kw in info.get("matched_keywords", []):
                if kw not in keyword_to_urls:
                    keyword_to_urls[kw] = []
                keyword_to_urls[kw].append(url)
        
        # Construire un prompt structuré par mot-clé
        lines = ["# Rapport de veille par thématique\n"]
        for keyword, urls in keyword_to_urls.items():
            lines.append(f"\n## Thématique: {keyword}\n")
            for url in urls:
                info = new_details.get(url, {})
                t = info.get("title") or "Sans titre"
                s = info.get("summary") or "Pas de résumé disponible"
                lines.append(f"- {t}\n  {url}\n  {s}\n")
        
        report_prompt = (
            "Rédige un rapport synthétique (en français) sur les actualités suivantes, "
            "organisé par thématique. Pour chaque thématique, résume les points clés "
            "et termine par une synthèse globale avec 2-3 recommandations actionnables.\n\n"
            + "\n".join(lines)
        )
        report = call_gemini_with_retry(prompt=report_prompt)
        report_text = report or "\n".join(lines)

    # Step 5: Persist
    if new_urls:
        memory_seen = set(memory.get("seen_urls", []))
        memory_seen.update(new_urls)
        memory["seen_urls"] = list(memory_seen)

        merged_details = memory.get("details", {})
        merged_details.update(new_details)
        memory["details"] = merged_details

        if report_text:
            import datetime
            report_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "new_urls": list(new_urls),
                "report": report_text,
            }
            reports_list = memory.get("reports", [])
            reports_list.append(report_entry)
            memory["reports"] = reports_list
            
            # Envoyer le rapport par email
            send_report_via_email(
                subject=f"Rapport de veille - {len(new_urls)} nouvelles actualités",
                body=report_text
            )

        atomic_save_memory(memory)
    else:
        logger.info("Aucun nouveau contenu pertinent trouvé.")
    logger.info("Tâche de veille terminée.")


# -----------------------------------------------------------------------------
# FastAPI app
app = FastAPI(
    title="Watcher API v2 (with Scraping & URL Context)",
    description="API de veille stratégique avec scraping basique et contexte URL pour Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/watch", summary="Déclenche la veille en tâche de fond")
async def trigger_watch_endpoint(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(perform_watch_task)
    logger.info("Requête de veille reçue. Tâche programmée en arrière-plan.")
    return {
        "message": "La veille a été lancée en arrière-plan. Consultez les logs pour les détails."
    }


@app.get("/memory", summary="Affiche la mémoire complète")
async def get_memory_content() -> Dict[str, Any]:
    return safe_load_memory()


@app.get("/", summary="Endpoint de santé")
async def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Watcher API is operational."}


# Sources management
@app.get("/sources", summary="Lire la configuration des sources")
async def read_sources() -> Dict[str, Any]:
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = SourceConfig(**data)
        return config.dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Le fichier '{SOURCES_FILE}' est introuvable.")
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=500, detail=f"Le fichier '{SOURCES_FILE}' est invalide: {e}")


@app.post("/sources", summary="Modifier la configuration des sources")
async def update_sources(update: UpdateSourcesRequest) -> Dict[str, Any]:
    try:
        if os.path.exists(SOURCES_FILE):
            with open(SOURCES_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
            current_config = SourceConfig(**current_data)
        else:
            current_config = SourceConfig()
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=500, detail=f"Le fichier '{SOURCES_FILE}' est invalide: {e}")

    if update.replace is not None:
        new_config = update.replace
    else:
        new_config = SourceConfig(
            veille_par_sujet=list(current_config.veille_par_sujet),
            veille_par_url=list(current_config.veille_par_url),
        )
        if update.add_subjects:
            for subj in update.add_subjects:
                if subj not in new_config.veille_par_sujet:
                    new_config.veille_par_sujet.append(subj)
        if update.add_urls:
            for url in update.add_urls:
                if url not in new_config.veille_par_url:
                    new_config.veille_par_url.append(url)
        if update.remove_subjects:
            new_config.veille_par_sujet = [s for s in new_config.veille_par_sujet if s not in update.remove_subjects]
        if update.remove_urls:
            new_config.veille_par_url = [u for u in new_config.veille_par_url if u not in update.remove_urls]

    try:
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(new_config.dict(), f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture de '{SOURCES_FILE}': {e}")
    return new_config.dict()


# Details & reports
@app.get("/details", summary="Consulter les descriptions enregistrées")
async def get_details() -> Dict[str, Any]:
    memory = safe_load_memory()
    return memory.get("details", {})


@app.get("/reports", summary="Consulter l'historique des rapports générés")
async def get_reports() -> List[Dict[str, Any]]:
    memory = safe_load_memory()
    return memory.get("reports", [])

# -----------------------------------------------------------------------------
# Recipients management

@app.get("/recipients", summary="Obtenir la liste des destinataires")
async def get_recipients() -> List[str]:
    """
    Retourne la liste actuelle des adresses électroniques des destinataires. Si
    aucun destinataire n'est configuré, retourne une liste vide.
    """
    return safe_load_recipients()


@app.post("/recipients", summary="Modifier la liste des destinataires")
async def update_recipients(update: UpdateRecipientsRequest) -> List[str]:
    """
    Ajoute, supprime ou remplace la liste des destinataires. Les adresses sont
    dédupliquées et sauvegardées.
    """
    current = safe_load_recipients()
    if update.replace is not None:
        new_list = [str(addr).strip() for addr in (update.replace or []) if addr]
    else:
        new_list = list(current)
        if update.add:
            for addr in update.add:
                if addr and addr not in new_list:
                    new_list.append(addr)
        if update.remove:
            new_list = [addr for addr in new_list if addr not in update.remove]
    atomic_save_recipients(new_list)
    return new_list
