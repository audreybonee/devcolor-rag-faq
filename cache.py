import numpy as np

class QueryCache:
    """
    A semantic in-memory cache for queries.
    Stores embeddings of previous queries and returns the cached 
    response if a new query has high cosine similarity (e.g. > 0.85).
    """
    def __init__(self):
        # Store tuples of (query_embedding, response)
        self.cache = []

    def _cosine_similarity(self, vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get(self, query_embedding: np.ndarray, threshold: float = 0.95) -> str:
        for cached_embedding, response in self.cache:
            score = self._cosine_similarity(query_embedding, cached_embedding)
            if score >= threshold:
                return response
        return None

    def set(self, query_embedding: np.ndarray, response: str):
        self.cache.append((query_embedding, response))
