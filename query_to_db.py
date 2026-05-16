import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
def run_rag_system(query_text):
    print("\n starts rag ...")
    
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
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("retrieving and make response ...")
    response = rag_chain.invoke(query_text)
    
    print("\n response LLM : ")
    print(response)

if __name__ == "__main__":
    query = "What is the advantage of LLM Larg Language Model ?"
    run_rag_system(query)