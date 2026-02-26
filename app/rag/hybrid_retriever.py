from typing import List, Dict, Tuple, Any
from app.rag.embeddings import EmbeddingProvider
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import LLMReranker
from app.rag.db_client import shared_vstore

class HybridRetriever:
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.rewriter = QueryRewriter()
        self.reranker = LLMReranker()
        self.vstore = shared_vstore

    async def retrieve_context(self, original_query: str, top_k: int = 3, debug: bool = False) -> Tuple[str, List[Dict], Dict[str, Any]]:
        """Orchestrates the entire Hybrid Search + Reranking pipeline."""
        debug_info = {}
        
        # 1. Query Rewriting
        optimized_query = await self.rewriter.rewrite_query(original_query)
        if debug:
            debug_info["original_query"] = original_query
            debug_info["rewritten_query"] = optimized_query
            
        # 2. Embed the Optimized Query (Generates BOTH Dense & Sparse vectors)
        query_vectors = self.embedder.embed_query(optimized_query)
        
        # 3. Hybrid Search in Qdrant (using Reciprocal Rank Fusion)
        # We fetch a bit more (top_k * 2) to give the LLM reranker good candidates
        search_results = self.vstore.similarity_search(query_vectors, top_k=top_k * 2)
        
        raw_chunks = []
        for i, point in enumerate(search_results):
            raw_chunks.append({
                "id": i + 1,  # 1-based indexing for LLM citations
                "text": point.payload.get("text", ""),
                "filename": point.payload.get("filename", "Unknown Document"),
                "fusion_score": point.score # Qdrant's mathematical RRF combined score
            })
            
        if debug:
            debug_info["qdrant_hybrid_results"] = raw_chunks
            
        # 4. LLM Re-ranking
        ranked_chunks = await self.reranker.rerank(optimized_query, raw_chunks, top_n=top_k)
        
        if debug:
            debug_info["final_ranked_chunks"] = ranked_chunks

        # 5. Format the strict CONTEXT block for the final generation
        formatted_context = "CONTEXT:\n"
        for chunk in ranked_chunks:
            formatted_context += f"[{chunk['id']}] (Source: {chunk['filename']})\n{chunk['text']}\n\n"

        return formatted_context, ranked_chunks, debug_info