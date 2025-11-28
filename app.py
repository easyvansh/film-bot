# app.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

VECTOR_DB_PATH = "vectorstore"

app = FastAPI(title="RAG Film Theory Chatbot")


class Question(BaseModel):
    question: str


def get_vectorstore():
    # Same local embeddings used in ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    return vectordb


print("[APP] Loading vector store...")
vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
print("[APP] Vector store loaded.")


# 🔹 LLM for generation (still OpenAI, but only for answering, not embeddings)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)

# 🔹 Prompt template (no chains, just format + invoke)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful film theory assistant. "
                "Use ONLY the given context from course readings to answer the question. "
                "If the context doesn't contain the answer, say you don't know. "
                "Keep the answer concise and academically useful.\n\n"
                "Context:\n{context}"
            ),
        ),
        ("human", "{question}"),
    ]
)


@app.post("/ask")
def ask(question: Question):
    """
    Ask a film theory question.
    """
    # 1️⃣ Retrieve docs
    docs = retriever.invoke(question.question)

    # 2️⃣ Build context string
    context_text = "\n\n".join(
        f"[Source: {d.metadata.get('source')} | page {d.metadata.get('page')}] \n{d.page_content}"
        for d in docs
    )

    # 3️⃣ Format prompt and call LLM
    messages = prompt.format_messages(
        context=context_text,
        question=question.question,
    )
    response = llm.invoke(messages)
    answer = response.content

    # 4️⃣ Return sources
    sources = [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
        }
        for d in docs
    ]

    return {
        "question": question.question,
        "answer": answer,
        "sources": sources,
    }


@app.get("/")
def root():
    return {
        "message": "RAG Film Theory Chatbot is running. POST to /ask with {'question': '...'}"
    }
