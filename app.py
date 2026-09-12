import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖")
st.title("🤖 IEEE RAS RAG Assistant")

# Initialize and cache vector database
@st.cache_resource
def setup_vectorstore():
    loader = TextLoader("ieee_ras_info.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Sidebar for Key
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

query = st.text_input("Ask a question about IEEE RAS:")

if query:
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    if api_key.strip():
        try:
            # 2. Synthesize dynamic answer using Groq LLM
            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0,
                groq_api_key=api_key.strip()
            )
            
            prompt = (
                f"You are an AI assistant for IEEE RAS. Answer the question concise and accurately using ONLY the context provided.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            
            with st.spinner("Generating answer..."):
                response = llm.invoke(prompt)
                st.subheader("Answer:")
                st.write(response.content)
                
        except Exception as e:
            st.error(f"API Error: {e}")
            st.subheader("Retrieved Context (Fallback):")
            for doc in docs:
                st.info(doc.page_content)
    else:
        st.warning("Please enter your Groq API key in the sidebar for full AI generation. Displaying raw retrieved context below:")
        for doc in docs:
            st.info(doc.page_content)