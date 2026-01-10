import streamlit as st
import sqlite3
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Initialize LangChain with OpenAI
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")

# Connect to database and retrieve player info
def get_players():
    conn = sqlite3.connect("players.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, hits FROM topPlayers")
    players = cursor.fetchall()
    conn.close()
    
    docs = [Document(page_content=f"Player: {name}, Hits: {hits}") for name, hits in players]
    return docs

# Initialize FAISS for Retrieval-Augmented Generation (RAG)
def setup_rag():
    docs = get_players()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)  # Embedding for search
    return vectorstore

vectorstore = setup_rag()

# Streamlit UI
st.title("🏆 Baseball Player Stats (RAG-Powered)")
user_query = st.text_input("Ask about the top players:")

if user_query:
    retrieved_docs = vectorstore.similarity_search(user_query, k=4)
    context_text = "\n".join(doc.page_content for doc in retrieved_docs) or "No records found."

    prompt = (
        "You are an expert baseball analyst. Answer the user's question using the provided player data."
        " If the answer cannot be determined from the data, say so.\n\n"
        f"Player data:\n{context_text}\n\nQuestion: {user_query}\nAnswer:"
    )

    response = llm.invoke(prompt)

    st.write("🔍 **AI Response:**")
    st.write(response.content.strip())
