import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore

from app.config import Config

def get_loader(file_path):
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdf": return PyPDFLoader(file_path)
    if ext in ("txt", "md"): return TextLoader(file_path)
    raise ValueError(f"Unsupported: .{ext}")

def ingest_document(file_path):
    loader = get_loader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=Config.PINECONE_INDEX
    )

    return {
        "file": os.path.basename(file_path),
        "chunks": len(chunks),
        "pages": len(documents)
    }
