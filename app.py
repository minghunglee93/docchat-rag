"""
DocChat RAG API
FastAPI application for document Q&A
"""

import logging
import os
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from rag_engine import RAGEngine
from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="DocChat RAG API",
    description="AI-powered document Q&A using Retrieval Augmented Generation",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_engine = None
upload_dir = Path("./data/uploads")
upload_dir.mkdir(parents=True, exist_ok=True)


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(4, ge=1, le=10, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source documents")
    use_history: bool = Field(False, description="Use conversation history")


class QueryResponse(BaseModel):
    answer: str
    question: str
    sources: Optional[List[Dict]] = None
    num_sources: Optional[int] = None
    timestamp: str


class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks_added: int
    timestamp: str


class SystemInfoResponse(BaseModel):
    model: str
    total_documents: int
    unique_sources: int
    conversation_turns: int
    sources: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_engine

    logger.info("Initializing DocChat RAG system...")

    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found. Using HuggingFace embeddings.")
            embedding_model = "huggingface"
        else:
            embedding_model = "openai"

        # Initialize vector store
        vector_store = VectorStore(
            collection_name="docchat",
            embedding_model=embedding_model
        )

        # Initialize RAG engine
        rag_engine = RAGEngine(
            vector_store=vector_store,
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )

        logger.info("✓ DocChat RAG system ready!")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DocChat RAG API",
        "version": "1.0.0",
        "description": "AI-powered document Q&A system",
        "endpoints": {
            "POST /query": "Ask a question about documents",
            "POST /upload": "Upload a document",
            "POST /upload/url": "Add document from URL",
            "GET /documents": "List all documents",
            "GET /history": "Get conversation history",
            "DELETE /history": "Clear conversation history",
            "GET /info": "System information",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    return {
        "status": "healthy",
        "rag_engine_loaded": rag_engine is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about the documents

    - **question**: Your question
    - **k**: Number of relevant documents to retrieve (1-10)
    - **include_sources**: Include source documents in response
    - **use_history**: Use conversation history for context
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        logger.info(f"Query: {request.question}")

        # Query the RAG system
        result = rag_engine.query(
            question=request.question,
            k=request.k,
            include_sources=request.include_sources,
            use_history=request.use_history
        )

        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        response = QueryResponse(
            answer=result["answer"],
            question=result["question"],
            sources=result.get("sources"),
            num_sources=result.get("num_sources"),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Answer generated: {len(result['answer'])} chars")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT)

    - **file**: Document file to upload
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    # Check file extension
    allowed_extensions = {".pdf", ".docx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    try:
        # Save uploaded file
        file_path = upload_dir / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Uploaded file: {file.filename}")

        # Process and add to RAG system
        result = rag_engine.add_documents([str(file_path)])

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return DocumentUploadResponse(
            status="success",
            message=result["message"],
            filename=file.filename,
            chunks_added=result["added"],
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/url")
async def upload_from_url(url: str = Form(...)):
    """
    Add document from URL

    - **url**: URL to fetch document from
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        logger.info(f"Adding URL: {url}")

        # Process URL
        result = rag_engine.add_documents([url])

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return {
            "status": "success",
            "message": result["message"],
            "url": url,
            "chunks_added": result["added"],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the system"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        sources = rag_engine.vector_store.get_sources()
        count = rag_engine.vector_store.get_document_count()

        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    """Get conversation history"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        history = rag_engine.get_history()

        formatted_history = [
            {
                "question": q,
                "answer": a,
                "turn": i + 1
            }
            for i, (q, a) in enumerate(history)
        ]

        return {
            "history": formatted_history,
            "total_turns": len(formatted_history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history")
async def clear_history():
    """Clear conversation history"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        rag_engine.clear_history()

        return {
            "status": "success",
            "message": "Conversation history cleared",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", response_model=SystemInfoResponse)
async def system_info():
    """Get system information"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        info = rag_engine.get_system_info()

        return SystemInfoResponse(
            model=info["model"],
            total_documents=info["vector_store"]["total_documents"],
            unique_sources=info["vector_store"]["unique_sources"],
            conversation_turns=info["conversation_turns"],
            sources=info["vector_store"]["sources"]
        )

    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)