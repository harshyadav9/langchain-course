from langchain.tools import tool
from dotenv import load_dotenv
import os
import json
import requests
from tavily import TavilyClient
from rich import print
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re


load_dotenv()


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))



@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[dict]]:
    """Find up to five source candidates with titles, complete URLs and snippets."""
    results = tavily.search(query=query, max_results=1)
    candidates = [
        {"title": r.get("title", ""), "url": r["url"], "snippet": r.get("content", "")}
        for r in results.get("results", []) if r.get("url")
    ]
    return json.dumps(candidates, ensure_ascii=False), candidates


@tool(response_format="content_and_artifact")
def scrape_url(url: str) -> tuple[str, dict]:
    """Extract readable content from a candidate URL; report extraction failures explicitly."""
    result = _extract_url(url)
    return json.dumps(result, ensure_ascii=False), result


def _extract_url(url: str) -> dict:
    def success(content: str) -> dict:
        return {"url": url, "status": "ok", "content": content[:5000],
                "truncated": len(content) > 5000}

    def failure(error: str) -> dict:
        return {"url": url, "status": "error", "content": "", "error": error}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        # ── Fetch page ─────────────────────────────────────
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # ──────────────────────────────────────────────────
        # Strategy 1 → trafilatura (BEST for articles/blogs)
        # ──────────────────────────────────────────────────
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )

        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return success(cleaned)

        # ──────────────────────────────────────────────────
        # Strategy 2 → readability
        # ──────────────────────────────────────────────────
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "lxml")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return success(cleaned)

        # ──────────────────────────────────────────────────
        # Strategy 3 → fallback full page extraction
        # ──────────────────────────────────────────────────
        soup = BeautifulSoup(html, "lxml")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        cleaned = re.sub(r'\s+', ' ', text)

        if len(cleaned.strip()) > 200:
            return success(cleaned)

        return failure("Could not extract meaningful content from the page.")

    except requests.exceptions.Timeout:
        return failure("Request timed out while scraping the URL.")

    except requests.exceptions.HTTPError as e:
        return failure(f"HTTP error occurred: {str(e)}")

    except Exception as e:
        return failure(f"Could not scrape URL: {str(e)}")

