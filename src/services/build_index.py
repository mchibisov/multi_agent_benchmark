from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path("data/documents")
INDEX_DIR = Path("indexes/faiss")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


documents = []

for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
    documents.append(
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "document_id": path.stem,
                "source": str(path),
            },
        )
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(documents)

chunk_numbers = {}

for chunk in chunks:
    document_id = chunk.metadata["document_id"]
    number = chunk_numbers.get(document_id, 0)

    chunk.metadata["chunk_id"] = f"{document_id}_chunk_{number:03d}"
    chunk_numbers[document_id] = number + 1

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = FAISS.from_documents(chunks, embeddings)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
vector_store.save_local(str(INDEX_DIR))

print("Documents:", len(documents))
print("Chunks:", len(chunks))
print("Vectors in FAISS:", vector_store.index.ntotal)

for chunk in chunks:
    print(chunk.metadata["chunk_id"], "-", len(chunk.page_content), "chars")
