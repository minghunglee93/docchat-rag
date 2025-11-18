# DocChat - AI Document Q&A 🤖

RAG (Retrieval Augmented Generation) system for chatting with your documents using LLMs.

## 🎯 Features

- **Upload Documents**: PDF, DOCX, TXT, or URLs
- **AI-Powered Q&A**: Ask questions, get answers with citations
- **Vector Search**: Semantic search using embeddings
- **Conversation Memory**: Context-aware responses
- **REST API**: FastAPI backend with auto-docs

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Add Documents

```bash
# Process documents via CLI
python rag_engine.py document.pdf another.pdf

# Or add via API after starting server
```

### 3. Run

```bash
# Start API
python app.py

# Visit http://localhost:8000/docs for interactive API
```

### 4. Query

**CLI:**
```bash
python rag_engine.py document.pdf
# Then ask questions interactively
```

**API:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "Summarize the key points"}
)
print(response.json()["response"])
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Ask a question |
| `/upload` | POST | Upload document |
| `/upload/url` | POST | Add from URL |
| `/documents` | GET | List documents |
| `/history` | GET | Conversation history |
| `/info` | GET | System info |
| `/docs` | GET | Interactive API docs |

## 🐳 Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Configuration

**Model Options:**
- `gpt-3.5-turbo` (default) - Fast, cheap
- `gpt-4` - Best quality
- Local LLMs via Ollama (free)

**Embeddings:**
- OpenAI (requires API key)
- Ollama (free, local)
- HuggingFace (free, local)

Edit in `app.py` or `rag_engine.py`.

## 💡 How It Works

```
1. Documents → Chunks (1000 tokens)
2. Chunks → Embeddings (vectors)
3. Store in ChromaDB
4. Question → Find similar chunks
5. LLM generates answer from chunks
```

## 📊 Example Usage

```python
from rag_engine import RAGEngine
from vector_store import VectorStore

# Initialize
vector_store = VectorStore()
rag = RAGEngine(vector_store)

# Add documents
rag.add_documents(["document.pdf"])

# Query
result = rag.query("What are the main topics?")
print(result["response"])
```

## 🧪 Testing

```bash
# Test document processing
python document_processor.py test.pdf

# Test vector store
python vector_store.py test.pdf

# Test RAG (interactive)
python rag_engine.py test.pdf
```

## 📈 Performance

- **Embeddings**: ~100ms per document
- **Query**: 1-3 seconds (depends on LLM)
- **Retrieval**: <100ms
- **Supports**: 1000+ documents

## 🎓 What You'll Learn

- ✅ RAG architecture
- ✅ Vector databases (ChromaDB)
- ✅ Embeddings & semantic search
- ✅ LLM integration (OpenAI)
- ✅ Prompt engineering
- ✅ Document processing
- ✅ FastAPI development

## 🚧 Troubleshooting

**No OpenAI key?** Use Ollama/HuggingFace embeddings (free):
```python
vector_store = VectorStore(embedding_model="ollama")
```

**Slow?** Reduce chunk size or use fewer documents for retrieval (k=2).

**API errors?** Check `.env` file has correct OPENAI_API_KEY.

## 🌟 Use Cases

- Company knowledge base
- Research paper Q&A
- Legal document analysis
- Technical documentation chat
- Customer support automation

## 📝 License

MIT License - Free for personal and commercial use

---

**Built to demonstrate modern LLM & RAG skills** 🚀