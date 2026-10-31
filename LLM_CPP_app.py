
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CPP RAG Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 CPP Document Q&A")
st.write("Ask questions about your CPP.txt document using Gemma 2B.")

# ---------------- LOAD DOCUMENT ----------------

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "CPP.txt"

if not FILE_PATH.exists():
    st.error(f"CPP.txt not found in: {BASE_DIR}")
    st.stop()


# ---------------- CREATE VECTOR DATABASE ----------------

@st.cache_resource
def create_database(file_path, modified_time):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db


# ---------------- LOAD OLLAMA MODEL ----------------

@st.cache_resource
def load_llm():
    return OllamaLLM(
        model="gemma2:2b",
        temperature=0.2
    )


try:
    with st.spinner("Loading document and vector database..."):
        db = create_database(
            str(FILE_PATH),
            FILE_PATH.stat().st_mtime
        )

    llm = load_llm()

    st.success("Document and Gemma model are ready!")

except Exception as e:
    st.error(
        "Could not initialize the application. "
        "Check your Ollama server, model, and installed packages."
    )
    st.exception(e)
    st.stop()


# ---------------- QUESTION INPUT ----------------

question = st.text_input(
    "Ask a question about CPP:",
    placeholder="What is CPP?"
)

if st.button("Generate Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching document and generating answer..."):

                # Retrieve relevant document chunks
                results = db.similarity_search(
                    question,
                    k=4
                )

                if not results:
                    st.warning("No relevant content found.")
                    st.stop()

                # Combine retrieved context
                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                # Prompt for RAG
                prompt = PromptTemplate.from_template(
                    """
You are a helpful assistant answering questions about CPP.

Answer the question using only the provided document context.
If the answer is not present in the context, clearly say:
"The answer was not found in the provided document."

Keep the answer clear, accurate, and easy to understand.
Do not invent facts.

Document Context:
{context}

Question:
{question}

Answer:
"""
                )

                chain = prompt | llm

                answer = chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )

            # Display generated answer
            st.subheader("Answer")
            st.write(answer)

            # Display retrieved sources
            with st.expander("View relevant document passages"):
                for i, doc in enumerate(results, start=1):
                    st.markdown(f"**Passage {i}**")
                    st.write(doc.page_content)

        except Exception as e:
            st.error("Failed to generate the answer.")
            st.write(
                "Check that Ollama is running and gemma2:2b is installed."
            )
            st.exception(e)
