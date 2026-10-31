import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="CPP RAG Q&A", page_icon="📚")

st.title("📚 CPP Document Q&A")
st.write("Ask questions about your CPP.txt document.")

# Automatically load CPP.txt from the same folder as this app
BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "CPP.txt"

if not FILE_PATH.exists():
    st.error(f"CPP.txt not found in: {BASE_DIR}")
    st.stop()


@st.cache_resource
def create_database(file_path, modified_time):
    # Load document
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS database
    db = FAISS.from_documents(chunks, embeddings)

    return db


try:
    with st.spinner("Loading CPP.txt and preparing search..."):
        db = create_database(
            str(FILE_PATH),
            FILE_PATH.stat().st_mtime
        )

    st.success("CPP.txt loaded successfully!")

except Exception as e:
    st.error("Could not load the document or create the database.")
    st.exception(e)
    st.stop()


# Ask questions
question = st.text_input(
    "Enter your question:",
    placeholder="What is CPP?"
)

if st.button("Ask Question", type="primary"):
    if question.strip():
        with st.spinner("Searching the document..."):
            results = db.similarity_search(question, k=4)

        st.subheader("Relevant Answers")

        if results:
            for i, doc in enumerate(results, start=1):
                st.markdown(f"### Result {i}")
                st.write(doc.page_content)
        else:
            st.info("No relevant content found.")
    else:
        st.warning("Please enter a question.")