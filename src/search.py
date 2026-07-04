import os
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class RAGSearch:
    def __init__(self, persist_directory: str = "faiss_store"):
        self.store = FaissVectorStore(persist_directory)
        self.store.load()
        # Initialize Groq LLM (defaults to llama-3.1-8b-instant and reads GROQ_API_KEY from environment)
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant documents and summarize them to answer the query using Groq LLM.
        """
        # 1. Query the vector store
        results = self.store.query(query, top_k=top_k)
        if not results:
            return "No relevant documents found to answer your query."

        # 2. Combine the retrieved content
        context = "\n\n".join([doc.page_content for doc in results])

        # 3. Formulate prompt
        system_content = (
            "You are a helpful and precise assistant. Use the following retrieved context to answer the user's question. "
            "If the context doesn't contain the answer, say that you don't know based on the context. Do not make things up.\n\n"
            f"Context:\n{context}"
        )
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query)
        ]

        # 4. Invoke LLM
        try:
            print(f"[DEBUG] Invoking Groq LLM for query: '{query}'")
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error invoking Groq LLM: {e}"
