"""Format retrieved sources for the model."""

import enum
from typing import List
from .models import MinimalSource


class ContextFormatter:
    """Formats retrieved sources into context text."""

    def __init__(self, max_chunks: int = 5, max_chunk_size: int = 2000):
        """
        Args:
            max_chunks: Maximum number of sources to include
            max_chunk_size: Maximum characters per chunk
        """
        self.max_chunk_size = max_chunk_size
        self.max_chunks = max_chunks

    def _get_chunk_text(self, source: MinimalSource, chunks: List) -> str:
        """Extract text from chunks for a specific source.

        Args:
            source: The source to find
            chunks: List of all chunks

        Returns:
            The chunk text, or empty string if not found
        """
        for chunk in chunks:
            if (
                chunk.file_path == source.file_path
                and chunk.first_character_index == source.first_character_index
            ):
                return chunk.text
        return ""

    def format(self, sources: List[MinimalSource], chunks: List) -> str:
        """Convert sources to formatted context string.

        Args:
            sources: List of retrieved sources
            chunks: List of all chunks

        Returns:
            Formatted context string
        """
        if not sources:
            return "No relevant context available."
        context_parts = []
        for i, source in enumerate(sources[: self.max_chunks], 1):
            text = self._get_chunk_text(source, chunks)
            if not text:
                continue
            if len(text) > self.max_chunk_size:
                text = text[: self.max_chunk_size] + "..."
            context_parts.append(f"[{i}] {source.file_path}\n{text}")
            if not context_parts:
                return "No relevant context available."
        return "\n---\n".join(context_parts)
