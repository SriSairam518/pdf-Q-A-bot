# pdf-Q-A-bot – AI PDF Question Answering System

pdf-Q-A-bot is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their content. It extracts text from the PDF, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector store, and retrieves relevant information to generate answers using **Llama 3 via Groq**.

## 🚀 Features

* Upload and process PDF documents
* Extract text using **PyMuPDF**
* Split documents using **Recursive Character Text Splitting**
* Generate text embeddings using **Hugging Face `all-MiniLM-L6-v2`**
* Store and retrieve document embeddings using **FAISS**
* Generate context-aware answers using **Llama 3 70B**
* Interactive web interface using **Streamlit**
* Maintains question-and-answer history during the session

## 🧠 RAG Pipeline

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Hugging Face Embeddings
     ↓
FAISS Vector Store
     ↓
Similarity Retrieval
     ↓
Relevant Context
     ↓
Llama 3 via Groq
     ↓
Generated Answer
```

## 🛠️ Technologies Used

* **Python**
* **LangChain**
* **RAG**
* **Llama 3**
* **Groq**
* **FAISS**
* **Hugging Face Embeddings**
* **PyMuPDF**
* **Streamlit**
* **python-dotenv**

## 📂 Project Structure

```text
pdf-Q-A-bot/
│
├── app.py              # Streamlit application and RAG pipeline
├── utils.py            # PDF extraction, chunking and vector store creation
└── README.md
```

## ⚙️ How It Works

### 1. PDF Text Extraction

The uploaded PDF is processed using **PyMuPDF (`fitz`)** to extract its text.

### 2. Text Chunking

The extracted text is divided into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

* Chunk size: `1000`
* Chunk overlap: `100`

The overlap helps preserve context between neighboring chunks.

### 3. Embedding Generation

Each text chunk is converted into a numerical vector using the Hugging Face:

```text
all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the document content.

### 4. Vector Storage

The generated embeddings are stored in a **FAISS vector store**, which enables similarity-based retrieval.

### 5. Question Retrieval

When a user asks a question, the system searches the vector store to retrieve the most relevant document content.

### 6. Answer Generation

The retrieved context is passed to **Llama 3 70B through Groq**, which generates an answer based on the retrieved document content.

## 🔑 Environment Setup

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

> **Important:** Never commit your `.env` file or API key to GitHub.

## ▶️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd pdf-Q-A-bot
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Example Use Cases

pdf-Q-A-bot can be used for:

* Asking questions about study materials
* Searching information within research papers
* Understanding technical documentation
* Querying reports and business documents
* Summarizing or retrieving information from large PDFs

## 🔮 Future Improvements

Possible improvements include:

* Support for multiple PDF documents
* Better conversational memory
* Retrieval evaluation and performance metrics
* Improved chunking strategies
* Reranking retrieved documents
* Source/page citations in generated answers
* Support for additional document formats
* Deployment as a production application

## 👨‍💻 Skills Demonstrated

**Generative AI:** RAG, LLMs, Text Embeddings, Semantic Search
**NLP:** Text Processing, Document Chunking
**Frameworks:** LangChain, Streamlit
**Vector Search:** FAISS
**Models:** Llama 3, Hugging Face Embeddings
**APIs:** Groq API
**Programming:** Python
