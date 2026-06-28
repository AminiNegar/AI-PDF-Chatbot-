# AI PDF Chatbot (RAG System)
https://aoqex2rlxv9kcxkkdp4smn.streamlit.app/
A production-ready Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and have context-aware conversations with them. Built using LangChain, Streamlit, ChromaDB, and Groq API.

## Features
- **PDF Processing**: Automatically extracts, splits, and processes PDF documents into manageable chunks.
- **Vector Lifecycle**: Generates semantic embeddings and stores them in a local Chroma vector database.
- **Advanced Retrieval**: Utilizes LangChain Expression Language (LCEL) to fetch the most relevant context matching user queries.
- **Contextual Generation**: Powers conversations using state-of-the-art Llama models via Groq API, strictly limiting answers to the provided document context to prevent hallucinations.
- **Clean Chat UI**: Offers an intuitive, interactive chatbot interface powered by Streamlit.

## Tech Stack
- **Frontend/UI**: Streamlit
- **RAG Framework**: LangChain (v0.3+)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Engine**: Groq Cloud API (`llama-3.3-70b-versatile`)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AminiNegar/AI-PDF-Chatbot-
cd AI-PDF-Chatbot-
```
### 2. Set Up Environment Variables
Open app_ui.py and replace the placeholder API key with your actual Groq API key:
```bash
os.environ["GROQ_API_KEY"] = "gsk_YOUR_ACTUAL_GROQ_API_KEY"
```
3. Configure System Packages (Optional)
If you want the local environment to inherit pre-installed global system packages (like PyTorch or Streamlit), ensure your .venv/pyvenv.cfg includes:
```bash
include-system-site-packages = true
```
How to Run
Launch the application directly using your python module wrapper:
``` bash
python -m streamlit run app_ui.py
```
Once running, open your browser to the local address provided (typically http://localhost:8501).

Usage Guide
### 1.Upload: Use the sidebar to upload any standard PDF document.

### 2.Process: Click the "Process and Analyze PDF" button to trigger text chunking and vector index creation.

### 3.Chat: Once successful, use the main chat input to query your document. The assistant will answer strictly based on the PDF contents.

Project Structure :
```bash
rag_pdf_chat/
│
├── app_ui.py          # Main Streamlit UI and RAG execution pipeline
├── query_to_db.py     # CLI-based RAG testing script
├── temp_pdf/          # Temporary directory for storing uploaded files
├── chroma_db/         # Local persistent vector store database
└── README.md          # Project documentation
```
