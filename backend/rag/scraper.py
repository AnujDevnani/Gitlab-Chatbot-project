"""
scraper.py — Crawl GitLab Handbook & Direction, return clean text chunks.

Strategy
--------
1. Fetch a set of well-known high-value URLs (fast, deterministic).
2. Optionally follow internal links up to MAX_PAGES total (depth=1).
3. Strip HTML → split on headings/paragraphs → yield TextChunk objects.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Generator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ─── Seed URLs ────────────────────────────────────────────────────────────────
SEED_URLS: list[str] = [
    # Handbook top-level sections
    "https://handbook.gitlab.com/handbook/values/",
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/engineering/",
    "https://handbook.gitlab.com/handbook/product/",
    "https://handbook.gitlab.com/handbook/marketing/",
    "https://handbook.gitlab.com/handbook/finance/",
    "https://handbook.gitlab.com/handbook/legal/",
    "https://handbook.gitlab.com/handbook/security/",
    "https://handbook.gitlab.com/handbook/hiring/",
    "https://handbook.gitlab.com/handbook/communication/",
    "https://handbook.gitlab.com/handbook/leadership/",
    "https://handbook.gitlab.com/handbook/company/",
    "https://handbook.gitlab.com/handbook/remote-work/",
    "https://handbook.gitlab.com/handbook/total-rewards/",
    "https://handbook.gitlab.com/handbook/business-ops/",
    # Direction pages
    "https://about.gitlab.com/direction/",
    "https://about.gitlab.com/direction/ai-ml/",
    "https://about.gitlab.com/direction/create/",
    "https://about.gitlab.com/direction/plan/",
    "https://about.gitlab.com/direction/verify/",
    "https://about.gitlab.com/direction/deploy/",
    "https://about.gitlab.com/direction/secure/",
    "https://about.gitlab.com/direction/monitor/",
]

ALLOWED_DOMAINS = {"handbook.gitlab.com", "about.gitlab.com"}
MAX_PAGES = 80          # cap total pages crawled
CHUNK_SIZE = 400        # target words per chunk
CHUNK_OVERLAP = 60      # word overlap between adjacent chunks
REQUEST_DELAY = 0.3     # seconds between requests (be polite)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GitLabHandbookBot/1.0; "
        "+https://github.com/your-repo)"
    )
}


@dataclass
class TextChunk:
    text: str
    url: str
    title: str
    section: str = ""
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        logger.warning("Fetch failed %s: %s", url, exc)
        return None


def _extract_text_blocks(html: str, url: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (page_title, [(section_heading, paragraph_text), ...])."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove nav / footer / sidebar noise
    for tag in soup.select("nav, footer, header, .sidebar, .toc, script, style, [role=navigation]"):
        tag.decompose()

    title = soup.find("h1")
    page_title = title.get_text(strip=True) if title else urlparse(url).path.strip("/").replace("/", " › ")

    blocks: list[tuple[str, str]] = []
    current_section = page_title
    current_paras: list[str] = []

    def flush():
        nonlocal current_paras
        if current_paras:
            blocks.append((current_section, " ".join(current_paras)))
            current_paras = []

    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return page_title, blocks

    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        if el.name in ("h1", "h2", "h3", "h4"):
            flush()
            current_section = el.get_text(strip=True)
        else:
            txt = el.get_text(" ", strip=True)
            if len(txt) > 40:
                current_paras.append(txt)
    flush()

    return page_title, blocks


def _sliding_window_chunks(words: list[str], size: int, overlap: int) -> list[str]:
    chunks = []
    step = size - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        chunk = words[i : i + size]
        if len(chunk) >= 30:
            chunks.append(" ".join(chunk))
        if i + size >= len(words):
            break
    return chunks


def _collect_internal_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        p = urlparse(href)
        if p.scheme in ("http", "https") and p.netloc in ALLOWED_DOMAINS:
            clean = href.split("#")[0].rstrip("/")
            links.append(clean)
    return list(set(links))


# ─── Public API ──────────────────────────────────────────────────────────────

def scrape_all(follow_links: bool = True) -> list[TextChunk]:
    """
    Crawl seed URLs (+ optional internal links) and return TextChunk list.
    """
    visited: set[str] = set()
    queue: list[str] = list(SEED_URLS)
    all_chunks: list[TextChunk] = []

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        url = url.split("#")[0].rstrip("/")
        if url in visited:
            continue
        visited.add(url)

        logger.info("[%d/%d] Scraping %s", len(visited), MAX_PAGES, url)
        html = _fetch(url)
        if not html:
            continue

        if follow_links and len(visited) < MAX_PAGES:
            for link in _collect_internal_links(html, url):
                if link not in visited and link not in queue:
                    queue.append(link)

        page_title, blocks = _extract_text_blocks(html, url)
        for section, text in blocks:
            words = text.split()
            if len(words) < 30:
                continue
            for chunk_text in _sliding_window_chunks(words, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append(TextChunk(
                    text=chunk_text,
                    url=url,
                    title=page_title,
                    section=section,
                ))

        time.sleep(REQUEST_DELAY)

    logger.info("Scraped %d pages → %d chunks.", len(visited), len(all_chunks))
    return all_chunks
