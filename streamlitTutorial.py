import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello, Streamlit!")
st.write(":Streamlit: This is a simple Streamlit app.")
st.text("Lets get started!")

#conditional logic with widgets

name = st.text_input("Enter your name:")
if st.button("Great"):
    st.success(f"Hello, {name}!")

#Display Data and charts
df = pd.DataFrame(np.random.randn(10, 5),columns =['A','B','C','D','E'])
st.line_chart(df)
st.bar_chart(df)

#File uploading and caching topics

upload_file = st.file_uploader("upload a csv",type='csv')
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)

st.title("Text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("**Bold** and *italic*, `Code` ,[Link](https://www.streamlit.io) ")
st.code("for i in range(5): print(i)",language='python')

st.text_input("What's your name?")
st.text_area("Write something...")
st.number_input("pick a number", min_value=0, max_value=100)
st.slider("choose a range",0, 100)
st.selectbox("Select a fruit", ["Apple", "Banana", " Mango"])
st.multiselect("Choose toppings", ["Cheese", "Tomato", "Olives"])
st.radio("Pick one", ["Option A", "Option B"])
st.checkbox("I agree to the terms")

if st.checkbox("Show details"):
    st.info("Here are more details...")

option = st.radio("Choose view", ["Show chart", "Show table"])
if option == "Show chart":
    st.write("Chart would appear here")
else:
    st.write("Table would appear here")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

    if submitted:
        st.success(f"Logged in as {username}!")

st.image("https://www.kaggle.com/static/images/new-site-logo-light.svg", caption="Sample Image", width="content")
st.video("https://youtu.be/DDjr6gXqtN8?si=WCiZQwOZQLEGdCUd", format="video/mp4", start_time=0)