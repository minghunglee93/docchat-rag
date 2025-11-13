"""
Test suite for DocChat RAG system
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os

# Set test environment
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

from app import app
from document_processor import DocumentProcessor
from vector_store import VectorStore

client = TestClient(app)


class TestDocumentProcessor:
    """Test document processing"""

    def test_init(self):
        """Test initialization"""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100

    def test_load_txt(self):
        """Test loading text file"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n" * 10)
            temp_path = f.name

        try:
            processor = DocumentProcessor()
            docs = processor.load_txt(temp_path)

            assert len(docs) > 0
            assert docs[0].page_content
            assert 'source' in docs[0].metadata
        finally:
            os.unlink(temp_path)

    def test_chunk_documents(self):
        """Test document chunking"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        from langchain.docstore.document import Document
        doc = Document(
            page_content="This is a test. " * 50,  # Long text
            metadata={"source": "test.txt"}
        )

        chunks = processor.chunk_documents([doc])

        assert len(chunks) > 1
        assert all('chunk' in chunk.metadata for chunk in chunks)
        assert all(len(chunk.page_content) <= 120 for chunk in chunks)  # Allow some overflow

    def test_get_document_stats(self):
        """Test statistics calculation"""
        processor = DocumentProcessor()

        from langchain.docstore.document import Document
        docs = [
            Document(page_content="Test 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test 2", metadata={"source": "test2.txt"})
        ]

        stats = processor.get_document_stats(docs)

        assert stats['total_documents'] == 2
        assert stats['total_words'] == 4
        assert stats['unique_sources'] == 2


class TestVectorStore:
    """Test vector store operations"""

    @pytest.fixture
    def temp_vector_store(self):
        """Create temporary vector store"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                collection_name="test_collection",
                persist_directory=tmpdir,
                embedding_model="huggingface"  # Use free local embeddings
            )
            yield store

    def test_init(self, temp_vector_store):
        """Test initialization"""
        assert temp_vector_store.vectorstore is not None
        assert temp_vector_store.get_document_count() == 0

    def test_add_documents(self, temp_vector_store):
        """Test adding documents"""
        from langchain.docstore.document import Document

        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test document 2", metadata={"source": "test2.txt"})
        ]

        ids = temp_vector_store.add_documents(docs)

        assert len(ids) == 2
        assert temp_vector_store.get_document_count() == 2

    def test_similarity_search(self, temp_vector_store):
        """Test similarity search"""
        from langchain.docstore.document import Document

        # Add test documents
        docs = [
            Document(page_content="Python programming language", metadata={"source": "python.txt"}),
            Document(page_content="JavaScript web development", metadata={"source": "js.txt"})
        ]
        temp_vector_store.add_documents(docs)

        # Search
        results = temp_vector_store.similarity_search("Python coding", k=1)

        assert len(results) <= 1
        if results:
            assert "Python" in results[0].page_content or "programming" in results[0].page_content

    def test_get_sources(self, temp_vector_store):
        """Test getting unique sources"""
        from langchain.docstore.document import Document

        docs = [
            Document(page_content="Test 1", metadata={"source": "file1.txt"}),
            Document(page_content="Test 2", metadata={"source": "file2.txt"}),
            Document(page_content="Test 3", metadata={"source": "file1.txt"})
        ]
        temp_vector_store.add_documents(docs)

        sources = temp_vector_store.get_sources()

        assert len(sources) == 2
        assert "file1.txt" in sources
        assert "file2.txt" in sources


class TestAPI:
    """Test API endpoints"""

    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_health(self):
        """Test health check"""
        response = client.get("/health")
        # May be 200 or 503 depending on initialization
        assert response.status_code in [200, 503]

    def test_query_validation(self):
        """Test query input validation"""
        # Empty question
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

        # Question too long
        response = client.post("/query", json={"question": "x" * 2000})
        assert response.status_code == 422

        # Invalid k value
        response = client.post("/query", json={"question": "test", "k": 20})
        assert response.status_code == 422

    def test_query_structure(self):
        """Test query response structure"""
        response = client.post(
            "/query",
            json={"question": "What is this?", "k": 3}
        )

        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "question" in data
            assert "timestamp" in data

    def test_upload_validation(self):
        """Test upload file validation"""
        # Create test file with wrong extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test.xyz", f, "application/octet-stream")}
                )

            assert response.status_code == 400
        finally:
            os.unlink(temp_path)

    def test_documents_endpoint(self):
        """Test documents listing"""
        response = client.get("/documents")

        if response.status_code == 200:
            data = response.json()
            assert "total_chunks" in data
            assert "unique_sources" in data
            assert "sources" in data

    def test_history_endpoints(self):
        """Test history endpoints"""
        # Get history
        response = client.get("/history")
        if response.status_code == 200:
            data = response.json()
            assert "history" in data
            assert "total_turns" in data

        # Clear history
        response = client.delete("/history")
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"

    def test_info_endpoint(self):
        """Test system info"""
        response = client.get("/info")

        if response.status_code == 200:
            data = response.json()
            assert "model" in data
            assert "total_documents" in data


class TestIntegration:
    """Integration tests"""

    def test_end_to_end(self):
        """Test complete pipeline"""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("The capital of France is Paris. Paris is known for the Eiffel Tower.")
            temp_path = f.name

        try:
            # Process document
            processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
            chunks = processor.process_file(temp_path)

            assert len(chunks) > 0

            # Add to vector store
            with tempfile.TemporaryDirectory() as tmpdir:
                store = VectorStore(
                    collection_name="test",
                    persist_directory=tmpdir,
                    embedding_model="huggingface"
                )
                store.add_documents(chunks)

                # Search
                results = store.similarity_search("What is the capital of France?", k=1)
                assert len(results) > 0
                assert "Paris" in results[0].page_content

        finally:
            os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_file_path(self):
        """Test handling of non-existent file"""
        processor = DocumentProcessor()

        with pytest.raises(Exception):
            processor.load_document("nonexistent.pdf")

    def test_empty_query(self):
        """Test empty query handling"""
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_invalid_url(self):
        """Test invalid URL handling"""
        response = client.post("/upload/url", data={"url": "not-a-valid-url"})
        # Should fail or return error
        assert response.status_code in [400, 500, 503]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])