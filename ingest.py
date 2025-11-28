# ingest.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = "data"
VECTOR_DB_PATH = "vectorstore"


def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            print(f"[INGEST] Loading {path}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    print(f"[INGEST] Loaded {len(docs)} documents (pages).")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Split into {len(chunks)} chunks.")
    return chunks


def create_vectorstore(chunks):
    # 🔹 Local, free embeddings – no OpenAI call
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    vectordb.persist()
    print("[INGEST] Vector store created and persisted.")
    return vectordb


if __name__ == "__main__":
    if not os.path.isdir(DATA_PATH):
        print(f"[INGEST] data/ folder not found at {os.path.abspath(DATA_PATH)}")
        raise SystemExit(1)

    docs = load_documents()
    if not docs:
        print("[INGEST] No PDF files found in 'data/' folder. Add PDFs and rerun.")
        raise SystemExit(1)

    chunks = split_documents(docs)
    if not chunks:
        print("[INGEST] No chunks created. Check if PDFs have readable text.")
        raise SystemExit(1)

    create_vectorstore(chunks)
    print("[INGEST] Done.")
