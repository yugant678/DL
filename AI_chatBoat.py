
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="Robotics RAG", page_icon="🤖")
st.title("🤖 Robotics RAG Assistant")

DOCS = "documents"
DB = "vectorstore"
os.makedirs(DOCS, exist_ok=True)

@st.cache_resource
def models():
    return (
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        OllamaLLM(model="gemma2:2b", temperature=0)
    )

def build_db():
    docs = []
    for f in os.listdir(DOCS):
        path = os.path.join(DOCS, f)
        if f.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif f.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

    if not docs:
        raise ValueError("No PDF or TXT documents found!")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120
    ).split_documents(docs)

    db = FAISS.from_documents(chunks, models()[0])
    db.save_local(DB)
    return db, len(chunks)

def answer_question(question, db):
    results = db.similarity_search_with_relevance_scores(question, k=5)
    results = [(d, s) for d, s in results if s >= 0.25]

    if not results:
        return "Insufficient information in the provided documents.", []

    context = "\n\n".join(
        f"Source: {os.path.basename(d.metadata.get('source', 'Unknown'))}, "
        f"Page: {d.metadata.get('page', 0) + 1}\n{d.page_content}"
        for d, s in results
    )

    prompt = f"""Answer using ONLY the document context below.
If evidence is missing, say you cannot find sufficient information.
If sources conflict, explain the conflict and cite both sources.
Do not invent facts.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""

    answer = models()[1].invoke(prompt)
    sources = [
        {
            "file": os.path.basename(d.metadata.get("source", "Unknown")),
            "page": d.metadata.get("page", 0) + 1,
            "excerpt": d.page_content[:250]
        }
        for d, s in results
    ]
    return answer, sources

with st.sidebar:
    st.header("Knowledge Base")
    files = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Save Documents"):
        if files:
            for f in files:
                with open(os.path.join(DOCS, os.path.basename(f.name)), "wb") as out:
                    out.write(f.getbuffer())
            st.success("Documents saved!")

    if st.button("Build Knowledge Base"):
        try:
            with st.spinner("Indexing documents..."):
                db, count = build_db()
            st.session_state.db = db
            st.success(f"Knowledge base ready! {count} chunks indexed.")
        except Exception as e:
            st.error(str(e))

if "db" not in st.session_state and os.path.exists(
    os.path.join(DB, "index.faiss")
):
    try:
        st.session_state.db = FAISS.load_local(
            DB, models()[0], allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.warning(f"Could not load index: {e}")

question = st.chat_input("Ask a robotics question...")

if question:
    st.chat_message("user").write(question)

    if "db" not in st.session_state:
        st.warning("Upload documents and build the knowledge base first.")
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding answers..."):
                    answer, sources = answer_question(
                        question, st.session_state.db
                    )
                st.write(answer)

                with st.expander("Supporting Sources"):
                    for s in sources:
                        st.markdown(f"**{s['file']} — Page {s['page']}**")
                        st.caption(s["excerpt"])
            except Exception as e:
                st.error(f"Error: {e}")
