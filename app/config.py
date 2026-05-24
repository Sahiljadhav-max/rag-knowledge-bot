import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL          = os.getenv("LLM_MODEL", "llama3-8b-8192")
    CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K              = int(os.getenv("TOP_K_RESULTS", 5))
    VECTOR_STORE_PATH  = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    UPLOAD_FOLDER      = os.getenv("UPLOAD_FOLDER", "./uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "knowl-index")