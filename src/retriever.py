from pathlib import Path
from typing import List
from tqdm import tqdm

from .indexing import Indexer
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults


class Retriever:
    """Handles retrieval of relevant snippets from the index."""

    def __init__(self, index_dir: Path):
        """Initialize the retriever with a loaded index."""
        self.indexer = Indexer.load(index_dir)

    def search_single(self, query: str, top_k: int = 5) -> List[MinimalSource]:
        """Search for a single query and return MinimalSource objects."""
        results = self.indexer.search(query, top_k)

        sources = []
        for chunk, score in results:
            sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
        return sources

    def search_dataset(
        self, dataset_path: Path, top_k: int = 5
    ) -> StudentSearchResults:
        """Search over a whole dataset of questions."""
        import json
        from .models import RagDataset

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = RagDataset(**data)
        search_results = []

        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            sources = self.search_single(q.question, top_k)
            search_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )

        return StudentSearchResults(search_results=search_results, k=top_k)
