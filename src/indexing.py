"""Index building and persistence for the RAG pipeline."""

from pathlib import Path
from typing import List, Optional
import pickle
from tqdm import tqdm
from .chunking import Chunker, Chunk
from .bm25 import BM25


class IndexerError(Exception):
    """Custom exception for Indexer errors."""

    pass


class Indexer:
    """Builds and persists search indices over a codebase."""

    def __init__(self, max_chunk_size: int = 2000):
        """Initialize the indexer.

        Args:
            max_chunk_size: Maximum chunk size in characters.
        """
        self.max_chunk_size = max_chunk_size
        self.chunker = Chunker(max_chunk_size)
        self.index: Optional[BM25] = None
        self.chunks: List[Chunk] = []

    def ingest(self, repo_root: Path) -> None:
        """Find all files, chunk them, build BM25 index."""
        if not repo_root.exists():
            raise FileNotFoundError(f"Repository root not found: {repo_root}")

        self.chunks = []
        extensions = {".py", ".md", ".txt", ".markdown"}

        for file_path in tqdm(list(repo_root.rglob("*")), desc="Indexing files"):
            if not file_path.is_file():
                continue
            if any(p.startswith(".") for p in file_path.parts):
                continue

            if file_path.suffix not in extensions:
                continue

            try:
                chunks = self.chunker.chunk_file(file_path, repo_root)
                self.chunks.extend(chunks)
            except Exception as e:
                # Just warn and continue with other files
                print(f"Warning: Could not index {file_path}: {e}")
                continue

        if not self.chunks:
            raise ValueError("No chunks were generated from the repository")

        print(f"Building BM25 index with {len(self.chunks)} chunks...")
        self.index = BM25(self.chunks)

    def persist(self, output_dir: Path) -> None:
        """Save index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to persist. Run ingest() first.")

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(output_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"✅ Index saved to {output_dir} ({len(self.chunks)} chunks)")

    @classmethod
    def load(cls, input_dir: Path) -> "Indexer":
        """Load saved index from disk."""
        index_path = input_dir / "bm25_index.pkl"
        chunks_path = input_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {input_dir}. Run 'index' command first."
            )

        indexer = cls()

        with open(index_path, "rb") as f:
            indexer.index = pickle.load(f)

        with open(chunks_path, "rb") as f:
            indexer.chunks = pickle.load(f)

        print(f"✅ Loaded index from {input_dir} ({len(indexer.chunks)} chunks)")
        return indexer

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search the index for the given query.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of (Chunk, score) tuples.
        """
        if self.index is None:
            raise ValueError("Index not built. Call ingest() or load() first.")

        if not query or not query.strip():
            return []

        return self.index.search(query, top_k)
