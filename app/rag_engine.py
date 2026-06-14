import os
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from app.config import Config

PROMPT_TEMPLATE = """You answer questions using only the context below.
If the answer isn't in the context, say so clearly.
Always mention which document you used.

Context: {context}
Chat history: {chat_history}
Question: {question}
Answer:"""

class RAGEngine:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.llm = ChatGroq(
            model=Config.LLM_MODEL,
            temperature=0.1,
            groq_api_key=Config.GROQ_API_KEY
        )
        self.vector_store = None
        self.chain = None
        self._load()

    def _load(self):
        try:
            pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            self.vector_store = PineconeVectorStore(
                index=pc.Index(Config.PINECONE_INDEX),
                embedding=self.embeddings
            )
            self._build_chain()
        except Exception as e:
            print(f"Vector store load warning: {e}")

    def _build_chain(self):
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": Config.TOP_K, "fetch_k": Config.TOP_K * 3}
        )
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5
        )
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=PROMPT_TEMPLATE
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    def reload(self):
        self._load()

    def query(self, question):
        if not self.chain:
            return {"answer": "No documents indexed yet. Upload a file first.",
                    "sources": []}
        result = self.chain({"question": question})
        seen, sources = set(), []
        for doc in result.get("source_documents", []):
            key = (doc.metadata.get("source", ""),
                   doc.metadata.get("page", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:150] + "..."
                })
        return {"answer": result["answer"], "sources": sources}

rag_engine = RAGEngine()
