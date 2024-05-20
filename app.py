from langchain.llms import OpenAI
import streamlit as st
from langchain import HuggingFaceHub
import os

# Load environment variables from .env file if needed
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
# Function to load OpenAI model and get response
def get_openai_response(question):
    
    # llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.6)
    llm_huggingface=HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B",model_kwargs={"temperature":0.6,"max_length":64})
    response=llm_huggingface(question)
    
    return response

# Initialize our Streamlit app
st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# If ask button is clicked
if submit and input_text:
    response = get_openai_response(input_text)
    st.subheader("The Response is")
    st.write(response)
