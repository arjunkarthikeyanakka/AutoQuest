import streamlit as st
import os
import tempfile
from integration import DataQuest
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="PDF Question Generator",
    page_icon="📝",
    layout="wide"
)

st.title("PDF Question Generator and Answering System")
st.markdown("Upload a PDF file to generate and answer questions based on its content.")

if "app" not in st.session_state:
    st.session_state.app = DataQuest()
    
if "collection" not in st.session_state:
    if not st.session_state.app.vecDB.setup_milvus():
        st.error("Failed to connect to Milvus database. Please make sure it's running.")
    else:
        st.session_state.app.vecDB.checkForCollection("rag_docs")
        embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        st.session_state.collection = st.session_state.app.vecDB.setup_enhanced_collection(embed_dim=768)
        st.session_state.embedder = embedder

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    
if "questions" not in st.session_state:
    st.session_state.questions = []
    
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None
    
if "answer" not in st.session_state:
    st.session_state.answer = ""

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                st.session_state.app.setPDFFile(pdf_path)
                
                text = st.session_state.app.readPDFFile()
                metadata = st.session_state.app.readPDFMetaData()
                headers = st.session_state.app.getPDFSectionHeaders()
                
                st.session_state.app.vecDB.clear_collection(st.session_state.collection)
                stats = st.session_state.app.vecDB.ingest_document_improved(
                    text, metadata, headers, st.session_state.collection
                )
                
                if stats:
                    st.success(f"PDF processed successfully! Ingestion stats: {stats}")
                    st.session_state.pdf_processed = True
                else:
                    st.error("Failed to process PDF")
    
    if st.session_state.pdf_processed:
        st.header("Question Generation")
        
        difficulty = st.selectbox(
            "Select difficulty level",
            ["Easy", "Medium", "Hard"],
            index=0
        )
        
        
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                text = st.session_state.app.readPDFFile()
                
                generated_questions = st.session_state.app.createQs(text, difficulty.lower(), 5)
                
                st.session_state.questions = st.session_state.app.createNonRepeatingQs(
                    generated_questions, 0.6
                )
                
                st.success(f"Generated {len(st.session_state.questions)} questions!")

if not st.session_state.pdf_processed:
    st.info("📌 Please upload a PDF file and click 'Process PDF' to begin.")
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload a PDF document** - The system will extract text and build a searchable index
    2. **Generate questions** - AI will generate questions based on the document content
    3. **Get answers** - The system will answer questions using information from the document
    """)
    
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generated Questions")
        if not st.session_state.questions:
            st.info("No questions generated yet. Use the sidebar to generate questions.")
        else:
            for i, question in enumerate(st.session_state.questions):
                if st.button(f"Q{i+1}: {question}", key=f"q_{i}"):
                    st.session_state.selected_question = question
                    
                    with st.spinner("Generating answer..."):
                        answer = st.session_state.app.M2.rag_query(
                            question, 
                            st.session_state.collection,
                            st.session_state.embedder
                        )
                        st.session_state.answer = answer
    
    with col2:
        st.subheader("Question & Answer")
        if st.session_state.selected_question:
            st.markdown(f"**Q: {st.session_state.selected_question}**")
            st.markdown("**A:**")
            st.markdown(st.session_state.answer)
        else:
            st.info("Select a question from the left to see its answer.")

    st.markdown("---")
    
    st.subheader("Ask your own question")
    custom_question = st.text_input("Enter your question about the document")
    
    if custom_question and st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            answer = st.session_state.app.M2.rag_query(
                custom_question, 
                st.session_state.collection,
                st.session_state.embedder
            )
            st.markdown(f"**Q: {custom_question}**")
            st.markdown("**A:**")
            st.markdown(answer)
    
    with st.expander("Document Information"):
        metadata = st.session_state.app.readPDFMetaData()
        st.json(metadata)
        
        st.subheader("Document Structure")
        headers = st.session_state.app.getPDFSectionHeaders()
        if headers:
            for i, header in enumerate(headers):
                st.markdown(f"- {header}")
        else:
            st.info("No section headers detected in this document.")

def cleanup():
    if "pdf_path" in locals():
        try:
            os.remove(pdf_path)
        except:
            pass

import atexit
atexit.register(cleanup)