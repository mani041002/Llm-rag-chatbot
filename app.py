import streamlit as st
from llm_backend import (
    extract_text,
    chunk_text,
    create_vector_store,
    retrieve_context,
    ask_llm
)

st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📄 Chat with your PDF")

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("❌ Could not extract text (scanned PDF not supported)")
    else:
        chunks = chunk_text(text)
        index, chunks = create_vector_store(chunks)

        if index is None:
            st.error("❌ Failed to process PDF")
        else:
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.messages = []
            st.success("✅ PDF processed! Ask questions below 👇")

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Ask something about your PDF...")

if user_input:
    if st.session_state.index is None:
        st.warning("⚠️ Please upload a PDF first")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve context
        context = retrieve_context(
            user_input,
            st.session_state.index,
            st.session_state.chunks
        )

        # Build prompt
        final_prompt = f"""
        Answer ONLY from the context below.
        If not found, say "Not found in document".

        Context:
        {context}

        Question:
        {user_input}
        """

        # Get response
        reply = ask_llm(final_prompt)

        # Show AI message
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)