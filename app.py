import streamlit as st
from pdf_processor import extract_text_from_pdf
from vector_db import VectorDB
from gemini_integration import GeminiAI

# Initialize only once
if "initialized" not in st.session_state:
    st.session_state.vector_db = VectorDB()
    st.session_state.gemini_ai = GeminiAI()
    st.session_state.chat_history = []
    st.session_state.current_question = ""
    st.session_state.initialized = True

vector_db = st.session_state.vector_db
gemini_ai = st.session_state.gemini_ai

# UI
st.title("PDF Q&A with Google Gemini AI")
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            vector_db.store_embeddings(pdf_text)
    st.sidebar.success("All PDFs processed and added to the knowledge base!")

# Display chat history
st.header("Chat")
for chat in st.session_state.chat_history:
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            <p style="font-size: 16px; font-weight: bold;">You: {chat['question']}</p>
            <p>Gemini: {chat['answer']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Input
question = st.text_input("Enter your question:", value=st.session_state.current_question, placeholder="Type your question here...")

if st.button("ASK"):
    if question:
        with st.spinner("Finding the answer..."):
            context_chunks = vector_db.query(question)
            if context_chunks:
                answer = gemini_ai.get_answer(question, context_chunks)
            else:
                answer = "This is beyond my knowledge."
        # Add to chat history
        st.session_state.chat_history.append({"question": question, "answer": answer})
        st.session_state.current_question = ""  # clear input
        st.rerun()  # safe now, since state is preserved
