from sentence_transformers import SentenceTransformer
from typing import List


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[list]:
        """
        Generate embeddings for multiple text chunks
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> list:
        """
        Generate embedding for a user query
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        return embedding

