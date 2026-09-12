import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖")
st.title("🤖 IEEE RAS RAG Assistant")

@st.cache_resource
def init_rag():
    # 1. Load and Chunk Data
    loader = TextLoader("ieee_ras_info.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)

    # 2. Free Vector Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Free Local Model (No API Keys Required)
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=250,
        temperature=0.1
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 4. Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on this context:\n{context}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = init_rag()

query = st.text_input("Ask a question about IEEE RAS:")
if query:
    with st.spinner("Generating answer..."):
        response = rag_chain.invoke({"input": query})
        st.subheader("Answer:")
        st.write(response["answer"])