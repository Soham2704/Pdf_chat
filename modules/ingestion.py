import os
import uuid
import fitz # PyMuPDF
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = "./storage/chroma_db"

def clean_text(text):
    """
    Minimal cleaning to keep text recognizable by the highlighter.
    Replaces newlines with spaces to form paragraphs.
    """
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ingest_pdf(uploaded_files):
    all_docs = []
    
    if not os.path.exists("temp_pdf"):
        os.makedirs("temp_pdf")
        
    for file in uploaded_files:
        file_path = os.path.join("temp_pdf", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
            
        # 1. RAW TEXT EXTRACTION (Page by Page)
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            cleaned_text = clean_text(text)
            
            if cleaned_text:
                all_docs.append(Document(
                    page_content=cleaned_text,
                    metadata={
                        "source_document": file.name,
                        "page_number": i + 1 
                    }
                ))

    # 2. CHUNKING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=[".", "!", "?", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(all_docs)

    # 3. Add Chunk IDs
    for split in splits:
        split.metadata["chunk_id"] = str(uuid.uuid4())[:8]

    # 4. EMBEDDING
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)