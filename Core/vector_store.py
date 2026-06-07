from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# dir to save everything
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "vector_db")
EMBEDDING_MODEL = "mistral-embed"
_embeddings = None

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = MistralAIEmbeddings(
            model = EMBEDDING_MODEL
        )

    return _embeddings

def build_vector_store(transcript: str, collection_name: str) -> Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content = chunk, metadata= {'chunk_index': i}) for i , chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding= embeddings,
        collection_name= collection_name,
        persist_directory= CHROMA_DIR
    )

    return vector_store

def load_vector_store(collection_name: str) -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name = collection_name,
        embedding_function= embeddings,
        persist_directory= CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store: Chroma, k: int = 6):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )
