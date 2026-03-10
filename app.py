import os
import streamlit as st

from ingestion.loader import load_document
from ingestion.preprocessing import clean_text
from ingestion.chunking import chunk_text

from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.generator import GroqGenerator


# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="NotebookLM-style AI", layout="wide")
st.title("📘 NotebookLM-style AI Assistant")
st.write("Upload documents and ask grounded questions with citations.")

# ------------------------------
# Session State
# ------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embedder" not in st.session_state:
    st.session_state.embedder = TextEmbedder()

# Chat history stored as List[dict] with role/content for agent compatibility
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# Store a short doc summary so the query rewriter has context
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# ------------------------------
# Sidebar: File Upload
# ------------------------------
st.sidebar.header("📂 Upload Document")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    all_chunks = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_text = load_document(file_path)
            cleaned_text = clean_text(raw_text)
            chunks = chunk_text(cleaned_text)

            labeled_chunks = [
                f"[Doc: {uploaded_file.name} | Chunk {i+1}] {chunk}"
                for i, chunk in enumerate(chunks)
            ]

            all_chunks.extend(labeled_chunks)

        embeddings = st.session_state.embedder.embed_texts(all_chunks)

        vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        vector_store.add_embeddings(embeddings, all_chunks)

        st.session_state.vector_store = vector_store
        st.session_state.chunks = all_chunks

        generator = GroqGenerator()

        # Generate and cache a short doc summary for the query rewriter agent
        st.session_state.doc_summary = generator.summarize_document(all_chunks[:10])

        # Generate suggested questions
        st.session_state.suggested_questions = generator.generate_questions(all_chunks)

        # Reset chat history on new upload
        st.session_state.chat_history = []

    st.sidebar.success(f"✅ Processed {len(all_chunks)} chunks from {len(uploaded_files)} file(s)")

# ------------------------------
# Sidebar: Suggested Questions
# ------------------------------
if st.session_state.suggested_questions:
    st.sidebar.subheader("💡 Suggested Questions")
    for i, q in enumerate(st.session_state.suggested_questions):
        if st.sidebar.button(q, key=f"sq_{i}"):
            st.session_state.selected_question = q
            st.rerun()

# ------------------------------
# Main Area
# ------------------------------
st.header("💬 Ask a Question")

question = st.text_input(
    "Enter your question:",
    value=st.session_state.selected_question
)

# Clear the selected question after it populates the input
if st.session_state.selected_question:
    st.session_state.selected_question = ""

col1, col2 = st.columns(2)

# ------------------------------
# Ask Question (Agentic RAG)
# ------------------------------
if col1.button("Get Answer"):
    if not st.session_state.vector_store:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🤖 Agent rewriting query and generating answer..."):
            generator = GroqGenerator()

            # --- Agentic Step: Query Rewriting ---
            # Pass full chat history (as dicts) + doc summary so the agent
            # can resolve references like "explain this" or "tell me more"
            rewritten_query = generator.rewrite_query(
                query=question,
                history=st.session_state.chat_history,
                doc_summary=st.session_state.doc_summary
            )

            # Embed the rewritten (expanded) query
            query_embedding = st.session_state.embedder.embed_query(rewritten_query)

            # Retrieve relevant chunks
            relevant_chunks = st.session_state.vector_store.similarity_search(
                query_embedding, top_k=5
            )

            # Generate grounded answer
            answer = generator.generate_answer(rewritten_query, relevant_chunks)

            # Store in chat history as dicts (compatible with rewrite_query agent)
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.subheader("📌 Answer")
        st.write(answer)

        # Show what the agent rewrote the query to
        if rewritten_query.strip().lower() != question.strip().lower():
            st.caption(f"🔄 **Agent rewrote your query to:** _{rewritten_query}_")

        with st.expander("📄 Retrieved Context"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")

# ------------------------------
# Summarize Document
# ------------------------------
if col2.button("Summarize Document"):
    if not st.session_state.chunks:
        st.warning("Upload a document first.")
    else:
        with st.spinner("Generating summary..."):
            generator = GroqGenerator()
            summary = generator.summarize_document(st.session_state.chunks)
            # Cache the full summary too
            st.session_state.doc_summary = summary

        st.subheader("📝 Document Summary")
        st.write(summary)

# ------------------------------
# Chat History (all previous turns, excluding the latest)
# ------------------------------
history = st.session_state.chat_history

# Build pairs from dict-based history
pairs = []
i = 0
while i < len(history) - 1:
    if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
        pairs.append((history[i]["content"], history[i + 1]["content"]))
        i += 2
    else:
        i += 1

# Show previous turns only (exclude the last pair — already shown above as current answer)
previous_pairs = pairs[:-1] if pairs else []

if previous_pairs:
    st.divider()
    st.subheader("🧠 Conversation History")
    for idx, (q, a) in enumerate(previous_pairs, 1):
        with st.expander(f"Q{idx}: {q[:80]}{'...' if len(q) > 80 else ''}"):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {a}")