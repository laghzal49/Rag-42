"""Command-line interface for the RAG pipeline."""

import fire
import json
import pickle
from pathlib import Path

from .indexing import Indexer
from .retriever import Retriever
from .answer import AnswerGenerator


class RAGCLI:
    """RAG Pipeline CLI commands."""

    def __init__(self):
        """Set up paths."""
        self.data_root = Path("data")
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build the index from the codebase."""
        print(f"🔨 Indexing with max_chunk_size={max_chunk_size}...")
        indexer = Indexer(max_chunk_size)
        indexer.ingest(self.raw_dir)
        indexer.persist(self.processed_dir)
        print(f"✅ Index saved to {self.processed_dir}")

    def search(self, query: str, k: int = 5) -> None:
        """Search for a single query."""
        retriever = Retriever(self.processed_dir)
        sources = retriever.search_single(query, k)

        print(f"\n🔍 Search results for: '{query}'\n")
        if not sources:
            print("No sources found.")
            return

        for i, source in enumerate(sources, 1):
            print(f"{i}. {source.file_path}")
            print(
                f"   Characters: {source.first_character_index}-{source.last_character_index}\n"
            )

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single question."""
        # 1. Get sources from retriever
        retriever = Retriever(self.processed_dir)
        sources = retriever.search_single(query, k)

        # 2. Load chunks
        with open(self.processed_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        # 3. Generate answer
        generator = AnswerGenerator()
        answer = generator.generate(query, sources, chunks)

        # 4. Print result
        print(f"\n❓ Question: {query}\n")
        print(f"📝 Answer: {answer}\n")
        print("📚 Sources:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source.file_path}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search over a whole dataset."""
        dataset_path = Path(dataset_path)
        save_dir = Path(save_directory)

        retriever = Retriever(self.processed_dir)
        results = retriever.search_dataset(dataset_path, k)

        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / dataset_path.name

        with open(output_path, "w") as f:
            json.dump(results.model_dump(), f, indent=2)

        print(f"✅ Saved search results to {output_path}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a dataset."""
        search_results_path = Path(student_search_results_path)
        save_dir = Path(save_directory)

        # Load chunks
        with open(self.processed_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        generator = AnswerGenerator()
        generator.answer_dataset(search_results_path, save_dir, chunks)


def main():
    """Entry point."""
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    main()
