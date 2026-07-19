from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "indexes" / "faiss"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RetrieverService:
    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def search(self, question: str) -> list[dict]:
        results = self.vector_store.similarity_search_with_score(
            question,
            k=self.top_k,
        )

        return [
            {
                "chunk_id": document.metadata["chunk_id"],
                "document_id": document.metadata["document_id"],
                "text": document.page_content,
                "distance_score": float(score),
            }
            for document, score in results
        ]
