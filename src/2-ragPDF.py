import streamlit as st
import fitz  # PyMuPDF for PDF parsing
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question using the provided context."
        " If the context does not contain the answer, reply that the information is unavailable.\n\n{context}",
    ),
    ("human", "{question}"),
])

def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def process_document(text):
    """Splits text into chunks and stores in a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]

    # Embedding model for vector storage
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

def chat_with_pdf(query, vector_store):
    """Uses LangChain with OpenAI API for RAG-based response."""
    relevant_docs = vector_store.similarity_search(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-5-mini")
    prompt_value = QA_PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(prompt_value.to_messages())

    return response.content

# Streamlit UI
st.title("Chat with a PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.success("PDF uploaded successfully!")
    text = extract_text_from_pdf(uploaded_file)
    vector_store = process_document(text)

    user_input = st.text_input("Ask something about the PDF:")
    if user_input:
        response = chat_with_pdf(user_input, vector_store)
        st.write("Response:", response)
