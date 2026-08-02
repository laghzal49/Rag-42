import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Self


from transformers import AutoTokenizer

from .chunking import Chunk


class BM25:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "bert-base-uncased",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks: list[Chunk] = chunks
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.k1 = k1
        self.b = b

        self.corpus_tokens = self._tokenize_chunks()
        self.N = len(self.corpus_tokens)

        self.doc_lens = [len(doc) for doc in self.corpus_tokens]
        total = sum(self.doc_lens)
        self.avgdl = total / self.N if self.N > 0 else 0.0

        self.doc_tf = self._calculate_tf()
        self.idf = self._calculate_idf()

    def _tokenize_chunks(self) -> list[list[str]]:
        return [self.tokenizer.tokenize(chunk.text) for chunk in self.chunks]

    def _calculate_tf(self) -> list[Counter[str]]:
        """Precompute term counts per doc, once, instead of on every score() call."""
        return [Counter(doc) for doc in self.corpus_tokens]

    def _calculate_idf(self) -> dict[str, float]:
        doc_counts: Counter = Counter()
        for doc in self.corpus_tokens:
            doc_counts.update(set(doc))
        idf = {}
        for word, n_q in doc_counts.items():
            idf[word] = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
        return idf

    def score(self, query_tokens: list[str], index: int) -> float:
        doc_len = self.doc_lens[index]
        if doc_len == 0 or self.avgdl == 0:
            return 0.0

        doc_tf = self.doc_tf[index]
        length_norm = 1 - self.b + self.b * (doc_len / self.avgdl)

        total_score = 0.0
        for term in query_tokens:
            tf = doc_tf.get(term)
            if not tf:
                continue
            idf_val = self.idf.get(term, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm
            total_score += idf_val * (numerator / denominator)

        return total_score

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        query_tokens = self.tokenizer.tokenize(query)
        scores = [self.score(query_tokens, i) for i in range(self.N)]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        return [(self.chunks[i], scores[i]) for i in top_indices]
