import os
import numpy as np
from openai import OpenAI
from chunker import Chunker

class Retriever:
    """
    Handles embeddings and vector similarity search.
    Uses OpenAI's text-embedding-3-small and in-memory numpy cosine similarity.
    """
    def __init__(self, corpus_path: str, threshold: float = 0.4):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.threshold = threshold

        # Load and parse chunks
        chunker = Chunker(corpus_path)
        self.chunks = chunker.get_chunks()

        # Embed the questions
        print("Initializing retriever (embedding corpus)...")
        self._embed_corpus()

    def get_embedding(self, text: str):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)

    def _embed_corpus(self):
        for chunk in self.chunks:
            # We embed the full_text (question + answer) to better match user queries semantically
            embedding = self.get_embedding(chunk["full_text"])
            chunk["embedding"] = embedding

    def _cosine_similarity(self, vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def search(self, query_embedding: np.ndarray, top_k: int = 2):

        scored_chunks = []
        for chunk in self.chunks:
            score = self._cosine_similarity(query_embedding, chunk["embedding"])
            scored_chunks.append((score, chunk))

        # Sort by descending score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # print("\nTop retrieval matches:")
        # for score, chunk in scored_chunks[:top_k]:
        #     print(f"{score:.3f} | {chunk['question']}")

        # Filter by threshold
        results = []
        for score, chunk in scored_chunks[:top_k]:
            if score >= self.threshold:
                results.append((score, chunk))

        return results
