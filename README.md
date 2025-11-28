
# 🎬 RAG Film Theory Chatbot

*A Retrieval-Augmented Generation system for answering film theory questions using course PDFs.*

---

## 📌 Overview

The **RAG Film Theory Chatbot** is a domain-specific question-answering system designed for film students, researchers, and cinephiles.
It uses a **Retrieval Augmented Generation (RAG)** pipeline to:

* **Embed and index** course PDF readings (Mulvey, Bordwell, Eisenstein, Kracauer, etc.)
* **Retrieve relevant excerpts** using vector search
* **Generate accurate, citation-aware answers** using OpenAI models
* **Expose a simple FastAPI endpoint** for querying the system

This project solves the core problem that generic LLMs hallucinate or lack access to academic readings.
By grounding answers in your PDFs, the chatbot gives **trustworthy**, **course-accurate**, and **citation-rich** responses.

---

## 🔧 Tech Stack

**Backend**

* FastAPI
* Python 3.11
* LangChain (v1+ using `langchain-classic`)
* OpenAI API (`gpt-4o-mini` + `text-embedding-3-small`)
* ChromaDB (local persistent vector store)

**Utilities**

* PyPDFLoader for PDF parsing
* dotenv for configuration
* Uvicorn (dev server)

---

## 📁 Project Structure

```
rag-film-chatbot/
│
├── app.py                # FastAPI app + RAG pipeline
├── ingest.py             # PDF → chunks → embeddings → Chroma vectorstore
│
├── data/                 # Your course PDFs go here
├── vectorstore/          # Auto-generated Chroma persistent DB
│
├── diagrams/             # Architecture diagrams / visuals (optional)
│
├── requirements.txt
├── README.md             # (this file)
└── .env                  # API keys (not committed)
```

---

## 🚀 Features (Completed So Far)

### ✅ 1. PDF Ingestion Pipeline

* Loads all PDFs from `data/`
* Splits into semantic chunks
* Embeds using `text-embedding-3-small`
* Saves vectors into a persistent ChromaDB

### ✅ 2. Retrieval-Augmented API

* `/ask` (POST) endpoint
* Retrieves top-k relevant chunks
* Runs them through a structured RAG prompt
* Returns:

  * `answer`
  * `sources` (filename + page numbers)
  * `question`

### ✅ 3. Modern LangChain v1 Architecture

Using new modules:

* `langchain_classic.chains`
* `create_retrieval_chain`
* `create_stuff_documents_chain`
* `ChatPromptTemplate`

### ✅ 4. Environment-based Config

* `.env` file stores OpenAI API key
* Works locally & portable for deployment

### ✅ 5. Swagger UI Documentation

* Browse at `http://localhost:8000/docs`
* Test endpoints instantly

---

## 🧪 Quickstart (Run Locally)

### 1. Clone the repository

```bash
git clone <repo-url>
cd rag-film-chatbot
```

### 2. Create a virtual environment (Windows)

```bash
py -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create `.env`:

```
OPENAI_API_KEY=your_key_here
```

### 5. Place your PDFs in `/data`

Examples:

```
data/
  ├── Mulvey_VisualPleasure.pdf
  ├── Bordwell_ClassicalNarration.pdf
  └── Kracauer_TheoryOfFilm.pdf
```

### 6. Build the vector store

```bash
python ingest.py
```

### 7. Run the server

```bash
python -m uvicorn app:app --reload
```

### 8. Test through Swagger

Visit:

```
http://127.0.0.1:8000/docs
```

Test using:

```json
{
  "question": "Explain Mulvey's concept of the male gaze."
}
```

---

## 🧠 How the System Works (Architecture)

1. **Ingestion**

   * PDFs → text pages → semantic chunks
   * Each chunk is embedded via `text-embedding-3-small`
   * Stored in ChromaDB (`vectorstore/`)

2. **Querying**

   * User asks: “What is the male gaze?”
   * Query is embedded & matched against vectorstore
   * Top chunks are passed into a structured prompt
   * LLM generates an answer grounded in retrieved context
   * Output returns answer + source citations

A diagram will be added once the architecture stabilizes.

---

## 📈 Planned Features (Roadmap)

### 🔜 Short Term

* Add HTML `/chat` UI (simple browser chatbox)
* Better chunk metadata (clean page numbers)
* Add logging of queries + retrieved sources
* Add custom prompts for:

  * theory definitions
  * film analysis
  * citation formatting

### 🔜 Mid Term

* Add reranking (e.g., Cohere Rerank)
* PDF preview links for each source
* Multi-agent analysis (e.g., Theory Agent + Context Agent)
* Add error handling for:

  * missing PDFs
  * failed embeddings
  * empty results

### 🔜 Long Term

* Deploy on Render / Fly.io
* Switch to external DB (Postgres + pgvector)
* Add user accounts + saved queries
* Add streaming responses

---

## 📝 Testing Plan

### ✔️ Functional Tests

* **Ingestion test:** verify chunks > 0
* **Retriever test:** verify retrieval returns context
* **API test:** POST `/ask` returns:

  * status 200
  * `answer` key exists
  * `sources` list length > 0

### ✔️ Failure Tests

* No PDFs → ingestion should gracefully exit
* Bad API key → error message
* Empty question → validation error

---

## 📚 Use Cases

* Film theory homework helper
* Citation-aware study buddy
* Essay-writing research tool
* Classroom demo for RAG systems
* Personal archive of all your course readings

---

## 👤 Author

**Vansh Singh**
University of Alberta – Computing Science × Film Studies
RAG systems · AI tools · Film theory

---
