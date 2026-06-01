from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os

class RagKnowledgeEngine:
    """
    RAG Semantic Knowledge Search Engine
    Automates text embedding generation, vector ingestion, and dynamic context retrieval.
    """
    def __init__(self, collection_name="knowledge_base"):
        self.collection_name = collection_name
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:") # Stateful memory vector DB
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def add_document(self, doc_id, text, metadata=None):
        vector = self.encoder.encode(text).tolist()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vector, "payload": {"text": text, "metadata": metadata}}]
        )
        print(f"Document {doc_id} successfully indexed.")

    def query(self, text, limit=2):
        vector = self.encoder.encode(text).tolist()
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit
        )
        return [res.payload["text"] for res in search_result]

if __name__ == "__main__":
    engine = RagKnowledgeEngine()
    engine.add_document(1, "Novalabs.in is an elite AI research institution designing premium financial products.")
    engine.add_document(2, "Quantum computing accelerates complex matrix operations in neural networks.")
    print("Search Results for 'Novalabs':", engine.query("What is Novalabs.in?"))
