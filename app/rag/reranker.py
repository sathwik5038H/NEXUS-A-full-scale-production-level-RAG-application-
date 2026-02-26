import json
from typing import List, Dict
from app.llm.gateway import LLMGateway

class LLMReranker:
    def __init__(self):
        self.gateway = LLMGateway()

    async def rerank(self, query: str, retrieved_chunks: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Uses the LLM to re-evaluate and sort the retrieved chunks based on relevance to the query.
        """
        if not retrieved_chunks:
            return []

        # Prepare the context blocks for the prompt
        chunk_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text += f"--- Chunk ID: {chunk['id']} ---\n{chunk['text']}\n\n"

        prompt = (
            "You are an expert relevance ranking engine.\n"
            f"Given the user query: '{query}'\n\n"
            "Evaluate the following document chunks and rank them from MOST relevant to LEAST relevant.\n"
            "If a chunk is completely irrelevant, do not include its ID in the final list.\n\n"
            f"{chunk_text}\n"
            "Return ONLY a JSON object in this exact format:\n"
            '{ "ranked_ids": [id1, id2, id3] }'
        )

        try:
            response = await self.gateway.model.generate_content_async(
                prompt,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            ranked_ids = result.get("ranked_ids", [])

            # Rebuild the list based on the LLM's preferred order
            ranked_chunks = []
            for rid in ranked_ids:
                # Find the chunk with this ID
                for chunk in retrieved_chunks:
                    if chunk["id"] == rid:
                        ranked_chunks.append(chunk)
                        break
            
            # Slice to the requested top N
            return ranked_chunks[:top_n]

        except Exception as e:
            print(f"⚠️ Reranking failed, returning original RRF order: {e}")
            return retrieved_chunks[:top_n]