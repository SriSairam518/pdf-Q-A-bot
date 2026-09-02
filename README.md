# NoteMate – Multi-PDF AI Question Answering Bot

NoteMate is a lightweight, fast, and user-friendly **Multi-PDF Question Answering system** built with Streamlit and powered by Groq AI models. Upload multiple PDF documents simultaneously and ask questions to get instant, context-aware answers citing source documents.

## 🚀 Features

* **Multi-PDF Upload**: Upload and query single or multiple PDF files at once.
* **Instant Text Extraction**: Uses **PyMuPDF (`pymupdf`)** for fast, reliable extraction.
* **Cross-Document Context Retrieval**: Uses TF-IDF cosine similarity search across all documents with source document tagging.
* **Fast Groq LLM Inference**: Leverages Groq's high-speed AI models (e.g. `groq/compound`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`).
* **Document Stats**: Displays individual document statistics and overall counts in the sidebar.
* **Native Streamlit Chat Interface**: Clean, interactive chat UI with full conversation history.

## 📁 Project Structure

```text
pdf-Q-A-bot/
├── app.py              # Streamlit Web Interface (Multi-PDF Upload & Chat)
├── utils.py            # PDF Extraction, Multi-Doc Chunking & Groq API Integration
├── requirements.txt    # Project Dependencies
└── README.md           # Documentation
```

## ⚙️ Quickstart

### 1. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

*(Or input your API key directly in the sidebar settings)*

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.
