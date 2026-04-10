"""
Web Search Service using DuckDuckGo.
Provides live internet context for Retrieval Augmented Generation.
"""

import logging
from typing import Optional

from ddgs import DDGS

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 4) -> str:
    """
    Search the web for the given query using duckduckgo-search.
    Returns a formatted string containing the top results for RAG context.
    """
    if not query or query.strip() == "":
        return ""

    logger.info(f"Searching web for: {query}")
    
    try:
        results_list = []
        with DDGS() as ddgs:
            # We use text search; you can grab up to max_results items.
            results = ddgs.text(query, max_results=max_results)
            for i, r in enumerate(results):
                title = r.get("title", "No Title")
                body = r.get("body", "No content")
                href = r.get("href", "")
                
                results_list.append(
                    f"[Source: Web Search - {title}]\n"
                    f"URL: {href}\n"
                    f"{body}"
                )

        if not results_list:
            return ""
            
        combined_text = "\n\n---\n\n".join(results_list)
        return combined_text

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Warning: Web search failed ({e}). Could not retrieve live information."
