"""
Web Search Service using DuckDuckGo.
Provides live internet context for Retrieval Augmented Generation.
"""

import logging
from typing import Optional

from ddgs import DDGS

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 4) -> list[dict]:
    """
    Search the web for the given query using duckduckgo-search.
    Returns a list of result dictionaries for RAG context.
    Each dict contains: 'title', 'body', 'href', 'source_type'.
    """
    if not query or query.strip() == "":
        return []

    logger.info(f"Searching web for: {query}")
    
    try:
        results_list = []
        with DDGS() as ddgs:
            # We use text search; you can grab up to max_results items.
            results = ddgs.text(query, max_results=max_results)
            for i, r in enumerate(results):
                results_list.append({
                    "title": r.get("title", "No Title"),
                    "body": r.get("body", "No content"),
                    "href": r.get("href", ""),
                    "source_type": "Web Search"
                })

        return results_list

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []
