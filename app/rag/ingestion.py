import io

import uuid

from datetime import datetime

from pypdf import PdfReader

from typing import Dict, Any

from app.rag.db_client import shared_vstore



from app.rag.chunking import ChunkingService

from app.rag.embeddings import EmbeddingProvider

from app.rag.vector_store import VectorStore



class IngestionService:

    def __init__(self):

        # We start with 500 tokens & 10% overlap.

        # (We will experiment with these numbers later!)

        self.chunker = ChunkingService(chunk_size=500, chunk_overlap=50)

        self.embedder = EmbeddingProvider()

        self.vstore = shared_vstore



    async def process_pdf(self, file_bytes: bytes, filename: str, owner: str = "system") -> Dict[str, Any]:

        print(f"📄 Processing PDF: {filename}...")

       

        # 1. Extract Text

        reader = PdfReader(io.BytesIO(file_bytes))

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:

                text += extracted + "\n"



        # 2. Chunk

        chunks = self.chunker.chunk_text(text)

        print(f"✂️ Sliced into {len(chunks)} semantic chunks.")


        # 3. Embed (In production, you'd batch this to save API calls)
        embeddings = [self.embedder.embed_document_chunk(chunk) for chunk in chunks]

        print("🧠 Generated mathematical embeddings.")



        # 4. Store

        document_id = str(uuid.uuid4())

        metadata = {

            "filename": filename,

            "upload_date": datetime.utcnow().isoformat(),

            "owner": owner

        }

       

        self.vstore.add_documents(

            document_id=document_id,

            chunks=chunks,

            embeddings=embeddings,

            metadata=metadata

        )

        print("💾 Saved to ChromaDB Vector Store.")



        return {

            "status": "success",

            "document_id": document_id,

            "chunks_added": len(chunks),

            "filename": filename

        }