import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ✅ LangChain v1+ imports
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

VECTOR_DB_PATH = "vectorstore"

app = FastAPI(title="RAG Film Theory Chatbot")


class Question(BaseModel):
    question: str


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    return vectordb



print("[APP] Loading vector store...")
vectordb = get_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
print("[APP] Vector store loaded.")

# 🔹 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 🔹 Prompt for RAG
system_prompt = (
    "You are a helpful film theory assistant. "
    "Use ONLY the given context from course readings to answer the question. "
    "If the context doesn't contain the answer, say you don't know. "
    "Keep the answer concise and academically useful.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 🔹 Document combination chain (LLM + prompt)
document_chain = create_stuff_documents_chain(llm, prompt)

# 🔹 Retrieval chain (retriever + doc chain)
rag_chain = create_retrieval_chain(retriever, document_chain)


@app.post("/ask")
def ask(question: Question):
    """
    Ask a film theory question.
    """
    result = rag_chain.invoke({"input": question.question})

    # LangChain v1 retrieval_chain returns keys: "answer" & "context"
    answer = result["answer"]
    docs = result["context"]

    source_info = [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
        }
        for d in docs
    ]

    return {
        "question": question.question,
        "answer": answer,
        "sources": source_info,
    }


@app.get("/")
def root():
    return {
        "message": "RAG Film Theory Chatbot is running. POST to /ask with {'question': '...'}"
    }
