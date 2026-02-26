from typing import List, Dict, Tuple

from app.rag.embeddings import EmbeddingProvider

from app.rag.vector_store import VectorStore

from app.rag.db_client import shared_vstore



class RetrieverService:

    def __init__(self):

        self.embedder = EmbeddingProvider()

        self.vstore = shared_vstore



    def retrieve_context(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:

        """Searches Qdrant and formats the results into a Context Block."""

       

        # 1. Turn query into numbers

        query_vector = self.embedder.generate_query_embedding(query)

       

        # 2. Search the database

        results = self.vstore.similarity_search(query_vector, top_k=top_k)

       

        # 3. Format into a strict CONTEXT block for the LLM

        formatted_context = "CONTEXT:\n"

        retrieved_chunks = []

       

        for i, point in enumerate(results):

            chunk_id = i + 1  # Use 1-based indexing for citations [1], [2]

            text = point.payload.get("text", "")

            filename = point.payload.get("filename", "Unknown Document")

           

            formatted_context += f"[{chunk_id}] (Source: {filename})\n{text}\n\n"

           

            # Save raw chunks so we can mathematically verify citations later

            retrieved_chunks.append({

                "id": chunk_id,

                "text": text,

                "score": point.score

            })

           

        return formatted_context, retrieved_chunks