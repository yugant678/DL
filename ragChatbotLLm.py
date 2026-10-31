import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama


st.title("C++ RAG Chatbot with Ollama")


#Load and Process Data
def load_vectorstore():
    #Load Document
    loader = TextLoader("cpp.txt",encoding="utf-8")
    documents = loader.load()
    
    #Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20)
    final_documents=text_splitter.split_documents(documents)
    
    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    #Vector Storing
    db = FAISS.from_documents(final_documents, embeddings)
    return db
db=load_vectorstore()

#Load LLM
llm = Ollama(model="gemma2:2b")

#Chat Interface
user_question = st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Thinking.."):
        docs=db.similarity_search(user_question)
        context= "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the wuestion using only the context below.
        
        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)
        