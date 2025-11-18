"""
Vector Store Module
Handles ChromaDB vector database operations
"""

import chromadb
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from typing import List, Dict, Optional


class VectorStore:
    """Manage vector database for RAG"""

    def __init__(
            self,
            collection_name: str = "documents",
            persist_directory: str = "./vector_db",
            embedding_model: str = "ollama"
    ):
        """
        Initialize vector store

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
            embedding_model: "openai" or "huggingface"
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
            print("Using OpenAI embeddings (text-embedding-ada-002)")
        elif embedding_model == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.embeddings = OllamaEmbeddings(
                model="llama2",
                base_url=base_url
            )
        else:
            # Use free local embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("Using HuggingFace embeddings (all-MiniLM-L6-v2)")

        # Initialize ChromaDB
        self.vectorstore = None
        self._load_or_create()

    def _load_or_create(self):
        """Load existing vectorstore or create new one"""
        try:
            # Try to load existing store
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

            # Check if it has documents
            count = self.get_document_count()
            if count > 0:
                print(f"✓ Loaded existing vector store: {count} documents")
            else:
                print("✓ Created new empty vector store")

        except Exception as e:
            print(f"Creating new vector store: {e}")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store

        Args:
            documents: List of Document objects

        Returns:
            List of document IDs
        """
        if not documents:
            print("No documents to add")
            return []

        try:
            print(f"Adding {len(documents)} documents to vector store...")

            # Add documents and get IDs
            ids = self.vectorstore.add_documents(documents)

            print(f"✓ Added {len(documents)} documents")

            return ids

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar Document objects
        """
        try:
            if filter:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)

            return results

        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Search with relevance scores

        Args:
            query: Search query
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        try:
            if filter:
                results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vectorstore.similarity_search_with_score(query, k=k)

            return results

        except Exception as e:
            print(f"Error in similarity search with scores: {e}")
            return []

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")

            # Recreate empty store
            self._load_or_create()

        except Exception as e:
            print(f"Error deleting collection: {e}")

    def get_document_count(self) -> int:
        """Get number of documents in store"""
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except:
            return 0

    def get_all_documents(self) -> List[Document]:
        """Get all documents from store"""
        try:
            # Get all documents
            collection = self.vectorstore._collection
            results = collection.get()

            documents = []
            if results and 'documents' in results:
                for i, doc_text in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if 'metadatas' in results else {}
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []

    def get_sources(self) -> List[str]:
        """Get list of unique sources in the vector store"""
        try:
            documents = self.get_all_documents()
            sources = set()

            for doc in documents:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])

            return sorted(list(sources))

        except Exception as e:
            print(f"Error getting sources: {e}")
            return []

    def search_by_source(self, source: str, k: int = 10) -> List[Document]:
        """
        Get documents from a specific source

        Args:
            source: Source identifier
            k: Maximum number of documents

        Returns:
            List of documents from that source
        """
        try:
            # Search with source filter
            results = self.similarity_search(
                query="",  # Empty query
                k=k,
                filter={"source": source}
            )
            return results

        except Exception as e:
            print(f"Error searching by source: {e}")
            return []

    def get_statistics(self) -> Dict:
        """Get vector store statistics"""
        try:
            count = self.get_document_count()
            sources = self.get_sources()

            stats = {
                "total_documents": count,
                "unique_sources": len(sources),
                "sources": sources,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }

            return stats

        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

    def print_statistics(self):
        """Print vector store statistics"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("VECTOR STORE STATISTICS")
        print("=" * 60)
        print(f"Collection: {stats.get('collection_name', 'Unknown')}")
        print(f"Total Documents: {stats.get('total_documents', 0)}")
        print(f"Unique Sources: {stats.get('unique_sources', 0)}")

        if stats.get('sources'):
            print("\nSources:")
            for source in stats['sources']:
                print(f"  - {source}")

        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from document_processor import DocumentProcessor

    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore(
        collection_name="test_collection",
        embedding_model="huggingface"  # Use free local embeddings for testing
    )

    # Show current statistics
    vector_store.print_statistics()

    # Process and add documents if provided
    if len(sys.argv) > 1:
        print("\nProcessing documents...")

        processor = DocumentProcessor()
        chunks = processor.process_multiple_files(sys.argv[1:])

        if chunks:
            # Add to vector store
            vector_store.add_documents(chunks)

            # Show updated statistics
            vector_store.print_statistics()

            # Test search
            print("\nTesting search...")
            query = input("Enter search query (or press Enter to skip): ").strip()

            if query:
                print(f"\nSearching for: '{query}'")
                print("-" * 60)

                results = vector_store.similarity_search_with_score(query, k=3)

                for i, (doc, score) in enumerate(results, 1):
                    print(f"\nResult {i} (score: {score:.4f}):")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Content: {doc.page_content[:200]}...")
                    print("-" * 60)
    else:
        print("\nUsage: python vector_store.py <file1> <file2> ...")
        print("This will process files and add them to the vector store.")