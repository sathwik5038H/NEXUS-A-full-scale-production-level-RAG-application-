import uuid
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, path: str = "./nexus_db"):
        self.client = QdrantClient(path=path)
        self.collection_name = "nexus_knowledge"
        
        # We define Named Vectors: One for Dense (Semantic), one for Sparse (Keyword)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=3072, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def add_documents(
        self, 
        document_id: str, 
        chunks: List[str], 
        embeddings: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ):
        """Saves chunks, hybrid vectors, and tracking metadata to Qdrant."""
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_uuid = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=chunk_uuid,
                    vector={
                        "dense": emb["dense"],
                        # Sparse vectors require this specific object mapping
                        "sparse": models.SparseVector(
                            indices=emb["sparse"]["indices"],
                            values=emb["sparse"]["values"]
                        )
                    },
                    payload={"text": chunk, "document_id": document_id, "chunk_index": i, **metadata}
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def similarity_search(self, query_embedding: Dict[str, Any], top_k: int = 5):
        """Finds the most relevant chunks using Hybrid Search and RRF."""
        
        # We use Qdrant's Query API with Prefetch for Hybrid Fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # 1. Dense (Semantic) Search
                models.Prefetch(
                    query=query_embedding["dense"],
                    using="dense",
                    limit=top_k * 2  # Over-fetch slightly to give the fusion algorithm more data
                ),
                # 2. Sparse (Keyword) Search
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_embedding["sparse"]["indices"],
                        values=query_embedding["sparse"]["values"]
                    ),
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            # 3. Combine scores using Reciprocal Rank Fusion (RRF)
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )
        
        return results.points if hasattr(results, 'points') else results