import streamlit as st
import os
from dotenv import load_dotenv
from utils import process_multiple_pdfs, get_relevant_context, answer_question

load_dotenv()

st.set_page_config(
    page_title="NoteMate - PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

# Title & Description
st.title("📄 NoteMate – Multi-PDF Question Answering Bot")
st.markdown("Upload one or multiple PDF documents and ask questions across all of them!")

# API Key configuration
api_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ Settings")
    if not api_key:
        api_key = st.text_input("Enter Groq API Key:", type="password", help="Get your key at https://console.groq.com")
        if not api_key:
            st.warning("Please provide a Groq API key to proceed.")
    else:
        st.success("✅ Groq API Key loaded!")

    st.markdown("---")
    st.header("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select one or multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = {"doc_stats": [], "chunks": [], "file_names": []}

# Process uploaded files
if uploaded_files:
    current_filenames = [f.name for f in uploaded_files]
    if st.session_state.pdf_store["file_names"] != current_filenames:
        with st.spinner("Processing uploaded PDF documents..."):
            try:
                doc_stats, all_chunks = process_multiple_pdfs(uploaded_files)
                st.session_state.pdf_store = {
                    "doc_stats": doc_stats,
                    "chunks": all_chunks,
                    "file_names": current_filenames
                }
                st.session_state.messages = []
                st.toast(f"Successfully loaded {len(doc_stats)} PDF document(s)!", icon="✅")
            except Exception as e:
                st.error(f"Failed to process PDF documents: {e}")
elif st.session_state.pdf_store["file_names"]:
    # User removed all uploaded files
    st.session_state.pdf_store = {"doc_stats": [], "chunks": [], "file_names": []}
    st.session_state.messages = []

# Sidebar Statistics
doc_stats = st.session_state.pdf_store.get("doc_stats", [])
chunks = st.session_state.pdf_store.get("chunks", [])

if doc_stats:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Document Overview")
        total_words = sum(d["word_count"] for d in doc_stats)
        st.write(f"**Total Documents:** {len(doc_stats)}")
        st.write(f"**Total Words:** {total_words:,}")
        st.write(f"**Total Chunks:** {len(chunks)}")

        with st.expander("📄 Loaded Files Detail"):
            for idx, doc in enumerate(doc_stats, 1):
                st.markdown(f"**{idx}. {doc['filename']}**")
                st.caption(f"Words: {doc['word_count']:,} | Chunks: {doc['chunk_count']}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Answer Generation
if prompt := st.chat_input("Ask a question about your uploaded PDF document(s)..."):
    if not api_key:
        st.error("Please enter a valid Groq API Key in the sidebar or `.env` file.")
    elif not chunks:
        st.warning("Please upload at least one readable PDF document first.")
    else:
        # Display user query
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    context = get_relevant_context(chunks, prompt)
                    answer = answer_question(context, prompt, api_key=api_key)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
