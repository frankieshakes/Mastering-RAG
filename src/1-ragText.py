import streamlit as st
import os
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the Streamlit page
st.set_page_config(page_title="Chat with a Variable", page_icon=":book:")
st.title("Basic LLM with some intelligence!")


# Define the string containing the details
details = """
Damian Montero is a 52 year old developer and AI fanatic with a teen aged daughter and a wife.
He's been married for 25 years and has a passion for technology.
This morning he had a bagel and a coffee for breakfast but currently he's in love with everything to do with AI.
"""

# Create a LangChain document from the string
document = Document(page_content=details)

# Load the document into a vector store
embeddings = OpenAIEmbeddings()  # Use OpenAI embeddings
vectorstore = FAISS.from_documents([document], embeddings)
retriever = vectorstore.as_retriever()

# Initialize the LLM
llm = OpenAI(temperature=0.7)  # Use OpenAI LLM

# Create a prompt template for QA
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

# Create the chain using LCEL
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Accept user queries
query = st.text_input("Ask a question about the details:")

if query:
    # Perform the query
    response = chain.invoke(query)
    st.write(f"**Answer:** {response}")