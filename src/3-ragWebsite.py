import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# Set your OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question using the provided website context."
        " If the answer cannot be found in the context, say you don't know.\n\nContext:\n{context}\n\n"
        "Conversation so far:\n{chat_history}",
    ),
    ("human", "{question}"),
])

def load_and_index_website(url):
    """Loads a website, splits it into chunks, and creates a FAISS index."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) # Use OpenAI embeddings
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def create_llm():
    """Initializes the ChatOpenAI model."""
    return ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")


def format_chat_history(messages):
    """Formats chat history into a plain-text transcript for prompting."""
    history_lines = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            speaker = "User"
        elif role == "assistant":
            speaker = "Assistant"
        else:
            continue
        history_lines.append(f"{speaker}: {message.get('content', '')}")

    return "\n".join(history_lines)


def generate_answer(question, vectorstore, llm, chat_history):
    """Generates an answer using retrieved context and conversation history."""
    retrieved_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt_value = QA_PROMPT.invoke(
        {"context": context, "chat_history": chat_history or "", "question": question}
    )
    response = llm.invoke(prompt_value.to_messages())
    return response.content

def main():
    st.title("Website Chatbot (OpenAI)")

    url = st.text_input("Enter the website URL:")

    if url:
        try:
            needs_reload = (
                st.session_state.get("loaded_url") != url
                or "vectorstore" not in st.session_state
                or "llm" not in st.session_state
            )

            if needs_reload:
                with st.spinner("Loading and indexing website..."):
                    vectorstore = load_and_index_website(url)
                st.session_state.vectorstore = vectorstore
                st.session_state.llm = create_llm()
                st.session_state.loaded_url = url
                st.session_state.messages = []

            st.success("Website loaded and indexed successfully!")

            messages = st.session_state.setdefault("messages", [])

            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about the website:"):
                messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                vectorstore = st.session_state.get("vectorstore")
                llm = st.session_state.get("llm")
                if vectorstore is None or llm is None:
                    raise RuntimeError("Vector store or language model not initialized.")

                history_text = format_chat_history(messages[:-1])

                full_response = ""
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking... ▌")
                    full_response = generate_answer(prompt, vectorstore, llm, history_text)
                    message_placeholder.markdown(full_response)

                messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

