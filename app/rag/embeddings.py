import os
import google.generativeai as genai
from typing import List, Dict, Any
from fastembed import SparseTextEmbedding

class EmbeddingProvider:
    def __init__(self):
        # 1. Setup Gemini for Dense Vectors (Semantic Meaning)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.dense_model = 'models/gemini-embedding-001'
        
        # 2. Setup FastEmbed for Sparse Vectors (Keyword/BM25)
        # This downloads a small local tokenizer to count term frequencies
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def generate_dense_embedding(self, text: str, task_type: str) -> List[float]:
        """Generates a semantic vector (dense) using Gemini."""
        result = genai.embed_content(
            model=self.dense_model,
            content=text,
            task_type=task_type
        )
        return result['embedding']

    def generate_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """Generates a keyword vector (sparse/BM25) using FastEmbed locally."""
        # FastEmbed returns a generator, so we cast to list and grab the first result
        sparse_vector = list(self.sparse_model.embed([text]))[0]
        
        # Qdrant expects sparse vectors strictly as a dict with 'indices' and 'values'
        return {
            "indices": sparse_vector.indices.tolist(),
            "values": sparse_vector.values.tolist()
        }

    def embed_document_chunk(self, text: str) -> Dict[str, Any]:
        """Returns BOTH dense and sparse vectors for a document chunk during ingestion."""
        return {
            "dense": self.generate_dense_embedding(text, task_type="retrieval_document"),
            "sparse": self.generate_sparse_embedding(text)
        }

    def embed_query(self, text: str) -> Dict[str, Any]:
        """Returns BOTH dense and sparse vectors for a user search query."""
        return {
            "dense": self.generate_dense_embedding(text, task_type="retrieval_query"),
            "sparse": self.generate_sparse_embedding(text)
        }