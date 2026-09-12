import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖")
st.title("🤖 IEEE RAS RAG Assistant")

@st.cache_resource
def load_rag_pipeline():
    # 1. Load Data
    loader = TextLoader("ieee_ras_info.txt")
    docs = loader.load()
    
    # 2. Chunk Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Vector Store with FREE HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

retriever = load_rag_pipeline()

# UI Query
query = st.text_input("Ask a question about IEEE RAS:")
if query:
    # Retrieve top relevant context chunks
    retrieved_docs = retriever.invoke(query)
    
    st.subheader("Retrieved Context & Answer:")
    for i, doc in enumerate(retrieved_docs):
        st.write(f"**Source Chunk {i+1}:**")
        st.info(doc.page_content)