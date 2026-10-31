from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Creating my prompts
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 'You are a helpful assistant. please respond to the question asked'),
        ("user", "Question:{question}")
    ]
)

#streamlit framework
st.title('Langchain demo with gemma2:2b')
input_text = st.text_input('what Question do you have in mind?')

#ollama model
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))