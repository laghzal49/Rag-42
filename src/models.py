"""Pydantic data models for the RAG pipeline."""

from typing import List, Union
from pydantic import BaseModel


class MinimalSource(BaseModel):
    """Represents a single source of information."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""

    question_id: str
    question: str


class AnsweredQuestion(BaseModel):
    """Represents an answered question."""

    question_id: str
    question: str
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Represents search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents search results with an answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represents search results for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represents search results with answers for a dataset."""

    search_results: List[MinimalAnswer]
    k: int
