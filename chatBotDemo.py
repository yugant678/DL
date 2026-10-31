import streamlit as st

#import os
#from dotenv import load_dotenv
# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#---------------------------------------------------------------------------
#Page Config
#---------------------------------------------------------------------------
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask questions about your C++ introduction")


#---------------------------------------------------------------------------
# Load Environment Variables
#---------------------------------------------------------------------------

@st.cache_resource
def load_vectorstore():
    loader = TextLoader("CPP.txt", encoding="utf-8")
    documents = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    #Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    db = FAISS.from_documents(final_documents, embeddings)
    return db

# Load Vector DB (only once)
db = load_vectorstore()

# User Input

query = st.text_input("Enter your question about C++:")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("Retrieved Context")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)