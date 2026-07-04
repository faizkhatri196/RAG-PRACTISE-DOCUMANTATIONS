import os
from typing import List, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FaissVectorStore:
    def __init__(self, persist_directory: str = "faiss_store"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = None

    def build_from_documents(self, documents: List[Any]):
        """
        Split documents, generate embeddings, and build the FAISS database.
        """
        print(f"[DEBUG] Building FAISS vector store from {len(documents)} documents...")
        # Text splitter to chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[DEBUG] Split into {len(chunks)} chunks.")
        
        # Build FAISS db
        self.db = FAISS.from_documents(chunks, self.embeddings)
        
        # Save locally
        os.makedirs(self.persist_directory, exist_ok=True)
        self.db.save_local(self.persist_directory)
        print(f"[DEBUG] FAISS vector store saved to {self.persist_directory}")

    def load(self):
        """
        Load the FAISS database from the local directory. If it doesn't exist, build it from the 'data' directory.
        """
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"[DEBUG] Loading FAISS vector store from {self.persist_directory}...")
            self.db = FAISS.load_local(
                self.persist_directory, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("[DEBUG] FAISS vector store loaded successfully.")
        else:
            print(f"[WARNING] Local vector store path {self.persist_directory} does not exist or is empty. Building it automatically...")
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            if docs:
                self.build_from_documents(docs)
            else:
                print("[ERROR] No documents found in data/ to build the FAISS store.")

    def query(self, query: str, top_k: int = 3) -> List[Any]:
        """
        Query the FAISS database.
        """
        if not self.db:
            print("[ERROR] FAISS database has not been loaded or built.")
            return []
        
        print(f"[DEBUG] Querying FAISS: '{query}' with top_k={top_k}")
        results = self.db.similarity_search(query, k=top_k)
        return results
