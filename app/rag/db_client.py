from app.rag.vector_store import VectorStore

# We initialize it ONCE here.
# Any other file that imports this will share the exact same open connection.
shared_vstore = VectorStore()