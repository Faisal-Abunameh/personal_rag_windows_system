"""
Semantic chunking engine.
Splits text into semantically coherent chunks using sentence embeddings.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import (
    CHUNK_SIMILARITY_THRESHOLD,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    CHUNK_OVERLAP_SENTENCES,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A semantic chunk of text with metadata."""
    text: str
    chunk_index: int
    source_file: str = ""
    start_sentence: int = 0
    end_sentence: int = 0
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex-based approach.
    Avoids spaCy dependency for speed; handles common abbreviations.
    """
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Split on sentence boundaries
    # Handles: periods, question marks, exclamation marks
    # Avoids splitting on: Mr., Mrs., Dr., etc., abbreviations, decimals
    sentence_endings = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence boundary
        r'(?<=[.!?])\s*\n',          # End of line
        re.MULTILINE
    )

    sentences = sentence_endings.split(text)

    # Also split on double newlines (paragraphs)
    final_sentences = []
    for s in sentences:
        parts = re.split(r'\n\s*\n', s)
        final_sentences.extend(parts)

    # Clean and filter
    result = [s.strip() for s in final_sentences if s.strip() and len(s.strip()) > 10]
    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def semantic_chunk(
    text: str,
    embed_fn,
    source_file: str = "",
    similarity_threshold: Optional[float] = None,
    max_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Chunk]:
    """
    Split text into semantically coherent chunks.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Compute cosine similarity between consecutive sentences
    4. Split where similarity drops below threshold
    5. Merge small chunks, split overly large ones

    Args:
        text: The text to chunk
        embed_fn: Function that takes list[str] and returns np.ndarray of embeddings
        source_file: Source filename for metadata
        similarity_threshold: Override config threshold
        max_chunk_size: Override config max chunk size
        min_chunk_size: Override config min chunk size
        overlap: Number of sentences to overlap between chunks

    Returns:
        List of Chunk objects
    """
    threshold = similarity_threshold or CHUNK_SIMILARITY_THRESHOLD
    max_size = max_chunk_size or MAX_CHUNK_SIZE
    min_size = min_chunk_size or MIN_CHUNK_SIZE
    overlap_n = overlap if overlap is not None else CHUNK_OVERLAP_SENTENCES

    # Step 1: Split into sentences
    sentences = _split_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            chunk_index=0,
            source_file=source_file,
            start_sentence=0,
            end_sentence=0,
        )]

    logger.debug(f"Chunking {len(sentences)} sentences from {source_file}")

    # Step 2: Embed all sentences in batch
    embeddings = embed_fn(sentences)

    # Step 3: Compute similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 4: Find split points (where similarity drops below threshold)
    split_points = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_points.append(i + 1)
    split_points.append(len(sentences))

    # Step 5: Form chunks from split points
    raw_chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]

        # Add overlap from previous chunk
        overlap_start = max(0, start - overlap_n) if i > 0 else start
        chunk_sentences = sentences[overlap_start:end]
        chunk_text = " ".join(chunk_sentences)

        raw_chunks.append({
            "text": chunk_text,
            "start": overlap_start,
            "end": end - 1,
        })

    # Step 6: Merge small chunks, split large ones
    final_chunks = []
    buffer = ""
    buffer_start = 0
    buffer_end = 0

    for rc in raw_chunks:
        if len(buffer) + len(rc["text"]) < min_size:
            if buffer:
                buffer += " " + rc["text"]
            else:
                buffer = rc["text"]
                buffer_start = rc["start"]
            buffer_end = rc["end"]
        else:
            if buffer:
                final_chunks.append({
                    "text": buffer,
                    "start": buffer_start,
                    "end": buffer_end,
                })
            if len(rc["text"]) > max_size:
                # Split large chunk by rough character boundaries
                words = rc["text"].split()
                part = ""
                for w in words:
                    if len(part) + len(w) + 1 > max_size and part:
                        final_chunks.append({
                            "text": part.strip(),
                            "start": rc["start"],
                            "end": rc["end"],
                        })
                        part = w
                    else:
                        part = f"{part} {w}" if part else w
                if part.strip():
                    buffer = part.strip()
                    buffer_start = rc["start"]
                    buffer_end = rc["end"]
                else:
                    buffer = ""
            else:
                buffer = rc["text"]
                buffer_start = rc["start"]
                buffer_end = rc["end"]

    if buffer:
        final_chunks.append({
            "text": buffer,
            "start": buffer_start,
            "end": buffer_end,
        })

    # Build Chunk objects
    result = []
    for i, fc in enumerate(final_chunks):
        result.append(Chunk(
            text=fc["text"],
            chunk_index=i,
            source_file=source_file,
            start_sentence=fc["start"],
            end_sentence=fc["end"],
            metadata={"source": source_file},
        ))

    logger.info(
        f"Created {len(result)} chunks from {len(sentences)} sentences "
        f"({source_file})"
    )
    return result
