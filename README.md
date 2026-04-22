# 📄 RAG AI Document Assistant

An AI-powered application that allows users to upload PDF documents and ask questions using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features
- Upload PDF documents
- Ask questions in natural language
- Context-aware answers using LLM
- Semantic search using FAISS vector database

---

## 🧠 Tech Stack
- Python
- LangChain
- FAISS
- OpenAI API
- Streamlit

---

## ⚙️ Architecture

User → Streamlit UI → LangChain Pipeline → FAISS Vector DB → LLM → Response

---

## 🔄 Workflow
1. Upload PDF
2. Extract and split text into chunks
3. Convert chunks into embeddings
4. Store in FAISS vector database
5. Retrieve relevant chunks based on query
6. Generate answer using LLM

---

## ▶️ Run Locally

```bash
pip install streamlit langchain openai faiss-cpu pypdf tiktoken
python -m streamlit run app.py
