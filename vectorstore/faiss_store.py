import faiss
import numpy as np
from typing import List


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim (int): Dimension of embeddings
        """
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Add embeddings and corresponding text chunks to FAISS
        
        Args:
            embeddings (np.ndarray): Shape (n, dim)
            chunks (List[str]): Original text chunks
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks size mismatch")

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k most similar text chunks
        
        Args:
            query_embedding (np.ndarray): Shape (dim,)
            top_k (int): Number of results
        
        Returns:
            List of relevant text chunks
        """
        query_embedding = np.array([query_embedding])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results
