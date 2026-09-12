import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖")
st.title("🤖 IEEE RAS RAG Assistant")

# 1. Initialize Vector Store
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

# 2. Sidebar for Free Gemini API Key
api_key = st.sidebar.text_input("Enter Gemini API Key (Free)", type="password")

query = st.text_input("Ask a question about IEEE RAS:")

if query:
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    if api_key.strip():
        try:
            genai.configure(api_key=api_key.strip())
            # Updated to current active production model
            model = genai.GenerativeModel("gemini-3.6-flash")
            
            prompt = (
                f"You are an assistant for IEEE RAS. Answer the user's question concisely using ONLY the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\nAnswer:"
            )
            
            with st.spinner("Generating answer..."):
                response = model.generate_content(prompt)
                st.subheader("Answer:")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"API Error: {e}")
            st.subheader("Retrieved Context Chunks (Fallback):")
            for doc in docs:
                st.info(doc.page_content)
    else:
        st.warning("Please enter a free Gemini API key in the sidebar for generated answers. Displaying raw context chunks below:")
        for doc in docs:
            st.info(doc.page_content)