import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def process_pdf(pdf_path) :

    print('loading pdf file ...') 
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f'pdf successfully loaded . number if pages =>{len(pages)}')

    print('chunking pdf ...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500 , 
        chunk_overlap = 100 , 
        length_function  = len
    )
    chunks = text_splitter.split_documents(pages)
    print(f'file was splitted to {len(chunks)} chunks ')
    if chunks :
        print('chunk 1 : ')
        print(chunks[0].page_content)
        print("-" * 30)
    return chunks

def save_to_vector_db(chunks) :
    print('convert chunk to embedding ...')
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    persist_directory = './chroma_db'
    vector_store = Chroma.from_documents(
        documents = chunks , 
        embedding = embedding_model , 
        persist_directory = persist_directory
    )
    print(f'all vectors savec in {persist_directory}')
    return vector_store

if __name__ == '__main__' :
    pdf_chunks = process_pdf('Sample.pdf')
    db = save_to_vector_db(pdf_chunks)