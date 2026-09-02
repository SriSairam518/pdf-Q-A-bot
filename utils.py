import pymupdf as fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import groq
import os


def extract_text_from_pdf(pdf_source):
    """
    Extract text content from a PDF file path or file stream/bytes.
    Returns: string containing extracted text.
    """
    if isinstance(pdf_source, (str, os.PathLike)):
        doc = fitz.open(pdf_source)
    else:
        pdf_bytes = pdf_source.read() if hasattr(pdf_source, "read") else pdf_source
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)

    text_chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if page_text.strip():
            text_chunks.append(page_text)

    return "\n\n".join(text_chunks)


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits long text into overlapping chunks.
    """
    if not text:
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current_chunk) + len(p) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(p) > chunk_size:
                start = 0
                while start < len(p):
                    end = min(start + chunk_size, len(p))
                    chunks.append(p[start:end])
                    if end == len(p):
                        break
                    start += chunk_size - overlap
                current_chunk = ""
            else:
                current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def process_multiple_pdfs(uploaded_files):
    """
    Processes a list of uploaded PDF files.
    Returns:
    - doc_stats: list of dicts with individual file metadata
    - all_chunks: list of dicts {"source": filename, "text": chunk_text}
    """
    doc_stats = []
    all_chunks = []

    for file_obj in uploaded_files:
        filename = getattr(file_obj, "name", "Uploaded Document")
        extracted_text = extract_text_from_pdf(file_obj)
        if extracted_text.strip():
            file_chunks = chunk_text(extracted_text)
            for c in file_chunks:
                all_chunks.append({"source": filename, "text": c})
            doc_stats.append({
                "filename": filename,
                "word_count": len(extracted_text.split()),
                "chunk_count": len(file_chunks)
            })

    return doc_stats, all_chunks


def get_relevant_context(chunks_with_metadata, question, top_k=8, max_context_chars=16000):
    """
    Retrieves the most relevant chunks across all uploaded PDF documents.
    Ensures per-document coverage + global relevance ranking.
    Formats context tagged with source document names.
    """
    if not chunks_with_metadata:
        return ""

    doc_sources = list(set(item["source"] for item in chunks_with_metadata))
    texts = [f"[Document: {item['source']}]\n{item['text']}" for item in chunks_with_metadata]
    total_length = sum(len(t) for t in texts)

    # If all combined text fits within context budget, send everything for 100% complete coverage
    if total_length <= max_context_chars or len(texts) <= top_k:
        return "\n\n---\n\n".join(texts)

    try:
        vectorizer = TfidfVectorizer().fit(texts + [question])
        question_vector = vectorizer.transform([question])

        selected_indices = set()

        # 1. Ensure at least top 1 chunk from each document is included for cross-doc queries
        per_doc_k = max(1, top_k // len(doc_sources))
        for doc in doc_sources:
            doc_indices = [i for i, item in enumerate(chunks_with_metadata) if item["source"] == doc]
            if not doc_indices:
                continue
            doc_texts = [texts[i] for i in doc_indices]
            doc_vectors = vectorizer.transform(doc_texts)
            sims = cosine_similarity(question_vector, doc_vectors).flatten()
            top_in_doc = sims.argsort()[-per_doc_k:][::-1]
            for idx in top_in_doc:
                selected_indices.add(doc_indices[idx])

        # 2. Add global top relevant chunks across all documents
        chunk_vectors = vectorizer.transform(texts)
        global_sims = cosine_similarity(question_vector, chunk_vectors).flatten()
        global_top = global_sims.argsort()[-top_k:][::-1]
        for idx in global_top:
            selected_indices.add(idx)

        sorted_indices = sorted(selected_indices)
        selected_texts = [texts[i] for i in sorted_indices]
        return "\n\n---\n\n".join(selected_texts)
    except Exception:
        return "\n\n---\n\n".join(texts[:top_k])


def answer_question(context, question, api_key, model="groq/compound"):
    """
    Queries Groq API with context from uploaded document(s) and user question.
    """
    client = groq.Groq(api_key=api_key)

    system_prompt = (
        "You are NoteMate, a helpful, accurate, and concise AI assistant.\n"
        "Your task is to answer the user's question based strictly on the provided PDF context from the uploaded document(s).\n"
        "When information comes from multiple documents, synthesize the information clearly and cite the source document names (e.g., [Document: filename]).\n"
        "If comparing documents or summarizing across documents, structure your answer clearly with sections or bullet points.\n"
        "If the answer cannot be found in the context, politely state that the information is not present in the document(s).\n"
        "Format your response clearly using markdown."
    )

    user_prompt = f"### Document Context:\n{context}\n\n### Question:\n{question}"

    candidate_models = [model, "groq/compound", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound-mini"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to generate answer from Groq API: {last_error}")