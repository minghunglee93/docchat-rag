"""
RAG Engine Module
Core Retrieval Augmented Generation implementation
"""

import os
from operator import add

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing import Annotated, List, Dict, Optional, TypedDict

from document_processor import DocumentProcessor
from vector_store import VectorStore


# ===== State Definition =====
class AgentState(TypedDict):
    """State for the RAG agent graph."""
    query: str
    chat_history: Annotated[List[BaseMessage], add]
    context: List[str]
    response: str

class RAGEngine:
    """Retrieval Augmented Generation Engine"""

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "llama2",
        temperature: float = 0.0,
        max_tokens: int = 500,
        llm_provider: str = "ollama"
    ):
        """
        Initialize RAG engine

        Args:
            vector_store: VectorStore instance
            model_name: LLM model to use
            temperature: Model temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
            llm_provider: "openai" or "ollama"
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_provider = llm_provider.lower()

        # Initialize LLM based on provider
        if self.llm_provider == "ollama":
            # Use Ollama (local LLMs)
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url
            )
            print(f"✓ Using Ollama model: {model_name} at {base_url}")
        elif "gpt" in model_name.lower() or self.llm_provider == "openai":
            # Use OpenAI
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✓ Using OpenAI model: {model_name}")
        else:
            # Fallback
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url
            )
            print(f"✓ Using model: {model_name}")

        # Create Graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build LangGraph workflow"""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from the vector store."""
        query = state["query"]

        if self.vector_store.get_document_count() == 0:
            # No documents in the store
            state["context"] = []
        else:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=3)
            state["context"] = [doc.page_content for doc in docs]

        return state

    def generate_node(self, state: AgentState) -> AgentState:
        """Generate a response using the LLM."""
        query = state["query"]
        context = state.get("context", [])
        chat_history = state.get("chat_history", [])

        # Create prompt
        if context:
            system_message = (
                "You are a helpful AI assistant. Use the following context to answer the user's question. "
                "If the context doesn't contain relevant information, say so and provide a general answer.\n\n"
                f"Context:\n{chr(10).join(context)}"
            )
        else:
            system_message = (
                "You are a helpful AI assistant. Answer the user's question based on your knowledge."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "chat_history": chat_history[-6:]  # Keep last 3 exchanges
        })

        state["response"] = response.content
        state["chat_history"] = [
            HumanMessage(content=query),
            AIMessage(content=response.content)
        ]

        return state

    def query(self, question: str, thread_id: str = "default") -> dict:
        """Query the RAG agent."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(
                {
                    "query": question,
                    "chat_history": [],
                    "context": [],
                    "response": ""
                },
                config=config
            )

            return {
                "response": result["response"],
                "context": result.get("context", []),
                "status": "success"
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "context": [],
                "status": "error"
            }

    def add_documents(self, file_paths: List[str]) -> Dict:
        """
        Add new documents to the system

        Args:
            file_paths: List of file paths or URLs

        Returns:
            Dictionary with status
        """
        try:
            # Process documents
            processor = DocumentProcessor()
            chunks = processor.process_multiple_files(file_paths)

            if not chunks:
                return {
                    "status": "error",
                    "message": "No documents were processed",
                    "added": 0
                }

            # Add to vector store
            ids = self.vector_store.add_documents(chunks)

            return {
                "status": "success",
                "message": f"Successfully added {len(chunks)} chunks from {len(file_paths)} files",
                "added": len(chunks),
                "files": file_paths
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "added": 0
            }

    def get_system_info(self) -> Dict:
        """Get information about the RAG system"""
        stats = self.vector_store.get_statistics()

        info = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_store": stats,
            "conversation_turns": len(self.conversation_history)
        }

        return info


# Example usage and CLI
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or export it:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("="*60)
    print("DocChat RAG System - Interactive Mode")
    print("="*60)

    # Initialize vector store
    print("\nInitializing vector store...")
    vector_store = VectorStore(
        collection_name="docchat",
        embedding_model="ollama"  # or "huggingface" for free local
    )

    vector_store.print_statistics()

    # Add documents if provided
    if len(sys.argv) > 1:
        print("\nAdding documents to system...")
        rag = RAGEngine(vector_store)
        result = rag.add_documents(sys.argv[1:])
        print(f"\n{result['message']}")
    else:
        print("\nNo documents provided. Using existing vector store.")

    # Check if we have documents
    if vector_store.get_document_count() == 0:
        print("\n⚠ Warning: No documents in vector store!")
        print("Add documents first:")
        print("  python rag_engine.py <file1> <file2> ...")
        sys.exit(1)

    # Initialize RAG engine
    print("\nInitializing RAG engine...")
    rag = RAGEngine(
        vector_store=vector_store,
        temperature=0.0
    )

    print("\n✓ RAG system ready!")
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'history' - show conversation history")
    print("  - 'clear' - clear history")
    print("  - 'info' - show system info")
    print("  - 'quit' - exit")
    print("="*60)

    # Interactive loop
    while True:
        try:
            question = input("\nYour question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if question.lower() == 'history':
                history = rag.get_history()
                if history:
                    print("\nConversation History:")
                    print("-"*60)
                    for i, (q, a) in enumerate(history, 1):
                        print(f"\n{i}. Q: {q}")
                        print(f"   A: {a[:200]}...")
                else:
                    print("No conversation history yet.")
                continue

            if question.lower() == 'clear':
                rag.clear_history()
                continue

            if question.lower() == 'info':
                info = rag.get_system_info()
                print("\nSystem Information:")
                print("-"*60)
                for key, value in info.items():
                    print(f"{key}: {value}")
                continue

            # Query the system
            print("\nSearching and generating answer...")
            result = rag.query(question)

            # Display answer
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(result["response"])

            # Display sources
            if "sources" in result and result["sources"]:
                print("\n" + "="*60)
                print("SOURCES:")
                print("="*60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. Relevance: {source['relevance_score']:.4f}")
                    print(f"   Source: {source['metadata'].get('source', 'Unknown')}")
                    if 'page' in source['metadata']:
                        print(f"   Page: {source['metadata']['page']}")
                    print(f"   Content: {source['content'][:150]}...")

            print("="*60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")