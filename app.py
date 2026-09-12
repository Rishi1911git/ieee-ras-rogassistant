import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖")
st.title("🤖 IEEE RAS RAG Assistant")

# Sidebar for Free Groq API Key
api_key = st.sidebar.text_input("Enter Groq API Key (Free)", type="password")

if api_key:
    # 1. Load and Chunk Data
    loader = TextLoader("ieee_ras_info.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)

    # 2. Vector Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. LLM Generation - Explicitly pass groq_api_key
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key=api_key.strip())

    # 4. Custom Prompt
    system_prompt = (
        "You are an assistant for IEEE RAS. Answer the user's specific question using ONLY "
        "the context provided below. Be concise and relevant.\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. UI Input
    query = st.text_input("Ask a question about IEEE RAS:")
    if query:
        response = rag_chain.invoke({"input": query})
        st.subheader("Answer:")
        st.write(response["answer"])
else:
    st.info("Please enter a free Groq API Key in the sidebar to run dynamic answers.")