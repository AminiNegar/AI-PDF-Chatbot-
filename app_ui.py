import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="AI PDF Chat (RAG)", page_icon="🏅", layout="centered")
st.title("🏅 AI PDF Chat (RAG System)")
st.write("Upload a PDF file and ask questions about its content.")

if not os.path.exists("temp_pdf"):
    os.makedirs("temp_pdf")

with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_button = st.button("Process and Analyze PDF")

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

if uploaded_file and process_button:
    with st.spinner("Extracting text and generating embeddings... Please wait."):
        file_path = os.path.join("temp_pdf", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        try:
            db.delete_collection()
        except Exception:
            pass
            
        Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./chroma_db")
        
        st.session_state.db_ready = True
        st.success("PDF analyzed successfully! You can now start chatting.")

if st.session_state.db_ready:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    
    template = """You are an expert assistant. Answer the user's question using ONLY the provided context below.
If you do not know the answer, say 'I cannot find the answer in the PDF.' Do not make up facts.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your PDF..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload and process a PDF file from the sidebar to start.")