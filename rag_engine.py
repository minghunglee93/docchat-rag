"""
RAG Engine Module
Core Retrieval Augmented Generation implementation
"""

from typing import List, Dict, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os

from document_processor import DocumentProcessor
from vector_store import VectorStore


class RAGEngine:
    """Retrieval Augmented Generation Engine"""

    def __init__(
            self,
            vector_store: VectorStore,
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.0,
            max_tokens: int = 500
    ):
        """
        Initialize RAG engine

        Args:
            vector_store: VectorStore instance
            model_name: LLM model to use
            temperature: Model temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM
        if "gpt" in model_name.lower():
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✓ Using OpenAI model: {model_name}")
        else:
            # For other models (Ollama, etc.)
            self.llm = OpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"✓ Using model: {model_name}")

        # Create QA chain
        self._setup_qa_chain()

        # Conversation history
        self.conversation_history = []

    def _setup_qa_chain(self):
        """Setup question-answering chain"""

        # Create custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source of your information when possible.

Context:
{context}

Question: {question}

Answer: Let me help you with that."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create chain
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt
        )

    def query(
            self,
            question: str,
            k: int = 4,
            include_sources: bool = True,
            use_history: bool = False
    ) -> Dict:
        """
        Query the RAG system

        Args:
            question: User question
            k: Number of documents to retrieve
            include_sources: Include source documents in response
            use_history: Use conversation history for context

        Returns:
            Dictionary with answer and sources
        """
        try:
            # Add history to question if requested
            if use_history and self.conversation_history:
                context_questions = "\n".join([
                    f"Previous Q: {q}\nPrevious A: {a}"
                    for q, a in self.conversation_history[-3:]  # Last 3 exchanges
                ])
                enhanced_question = f"{context_questions}\n\nCurrent Question: {question}"
            else:
                enhanced_question = question

            # Retrieve relevant documents
            docs_with_scores = self.vector_store.similarity_search_with_score(
                enhanced_question,
                k=k
            )

            if not docs_with_scores:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "question": question
                }

            # Extract documents and scores
            docs = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]

            # Get answer from LLM
            result = self.qa_chain({
                "input_documents": docs,
                "question": question
            })

            answer = result["output_text"]

            # Store in conversation history
            self.conversation_history.append((question, answer))

            # Prepare response
            response = {
                "answer": answer,
                "question": question
            }

            # Add sources if requested
            if include_sources:
                sources = []
                for i, (doc, score) in enumerate(zip(docs, scores)):
                    source_info = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": float(score)
                    }
                    sources.append(source_info)

                response["sources"] = sources
                response["num_sources"] = len(sources)

            return response

        except Exception as e:
            print(f"Error in query: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "question": question,
                "error": str(e)
            }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")

    def get_history(self) -> List[tuple]:
        """Get conversation history"""
        return self.conversation_history

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

    print("=" * 60)
    print("DocChat RAG System - Interactive Mode")
    print("=" * 60)

    # Initialize vector store
    print("\nInitializing vector store...")
    vector_store = VectorStore(
        collection_name="docchat",
        embedding_model="openai"  # or "huggingface" for free local
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
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )

    print("\n✓ RAG system ready!")
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'history' - show conversation history")
    print("  - 'clear' - clear history")
    print("  - 'info' - show system info")
    print("  - 'quit' - exit")
    print("=" * 60)

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
                    print("-" * 60)
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
                print("-" * 60)
                for key, value in info.items():
                    print(f"{key}: {value}")
                continue

            # Query the system
            print("\nSearching and generating answer...")
            result = rag.query(question, k=3, include_sources=True)

            # Display answer
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(result["answer"])

            # Display sources
            if "sources" in result and result["sources"]:
                print("\n" + "=" * 60)
                print("SOURCES:")
                print("=" * 60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"\n{i}. Relevance: {source['relevance_score']:.4f}")
                    print(f"   Source: {source['metadata'].get('source', 'Unknown')}")
                    if 'page' in source['metadata']:
                        print(f"   Page: {source['metadata']['page']}")
                    print(f"   Content: {source['content'][:150]}...")

            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")