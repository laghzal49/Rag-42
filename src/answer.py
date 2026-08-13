"""Answer generation orchestrator."""

from pathlib import Path
from typing import List
import json
from tqdm import tqdm

from .models import MinimalAnswer, StudentSearchResultsAndAnswer
from .context_formatter import ContextFormatter
from .promt import PromptBuilder
from .model import ModelManager


class AnswerGenerator:
    """Orchestrates answer generation using RAG pipeline."""

    def __init__(self):
        """Initialize all components."""
        self.model: ModelManager = ModelManager()
        self.prompt: PromptBuilder = PromptBuilder()
        self.context: ContextFormatter = ContextFormatter()

    def generate(
        self,
        question: str,
        sources: List,
        chunks: List,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> str:
        """Generate a single answer.

        Steps:
        1. Format context from sources (use ContextFormatter)
        2. Build prompt from question and context (use PromptBuilder)
        3. Generate answer (use ModelManager)
        4. Return answer
        """
        context = self.context.format(sources, chunks)
        prompt = self.prompt.build(question, context)
        answer = self.model.generate(
            prompt, temperature=temperature, max_tokens=max_tokens
        )
        return answer

    def answer_dataset(
        self,
        search_results_path: Path,
        save_directory: Path,
        chunks: List,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> StudentSearchResultsAndAnswer:
        """Generate answers for a dataset."""

        with open(search_results_path, "r") as f:
            data = json.load(f)
        from .models import StudentSearchResults

        search_results = StudentSearchResults(**data)

        answers = []
        for result in tqdm(search_results.search_results, desc="Generating answers"):
            answer = self.generate(
                question=result.question,
                sources=result.retrieved_sources,
                chunks=chunks,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=search_results.k
        )

        save_directory.mkdir(parents=True, exist_ok=True)
        output_dir = save_directory / search_results_path.name

        with open(output_dir, "w") as f:
            json.dump(output.model_dump(), f, indent=2)

        print(f"✅ Answers saved to {output_dir}")
        return output
