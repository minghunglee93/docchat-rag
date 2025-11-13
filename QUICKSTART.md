# DocChat RAG - Quick Reference

## ⚡ One-Command Start

```bash
# Setup and run
chmod +x quickstart.sh
./quickstart.sh
```

## 🚀 Manual Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: Add OPENAI_API_KEY

# 3. Test with CLI
python rag_engine.py document.pdf

# 4. Run API
python app.py

# 5. Run UI (optional)
python ui.py
```

## 📡 API Quick Examples

```bash
# Upload document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Ask question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'

# Add URL
curl -X POST "http://localhost:8000/upload/url" \
  -d "url=https://example.com/article"

# List documents
curl http://localhost:8000/documents

# View history
curl http://localhost:8000/history

# System info
curl http://localhost:8000/info
```

## 🐍 Python Usage

```python
from rag_engine import RAGEngine
from vector_store import VectorStore

# Initialize
vector_store = VectorStore()
rag = RAGEngine(vector_store)

# Add documents
rag.add_documents(["doc1.pdf", "doc2.pdf"])

# Query
result = rag.query("What are the key points?")
print(result["answer"])

# With sources
result = rag.query("Explain...", k=5, include_sources=True)
for source in result["sources"]:
    print(f"Source: {source['metadata']['source']}")
```

## 🧪 Testing

```bash
# Run all tests
pytest test_rag.py -v

# Run specific test
pytest test_rag.py::TestAPI::test_query_validation -v

# With coverage
pytest test_rag.py --cov=. --cov-report=html
```

## 🐳 Docker

```bash
# Build
docker build -t docchat-rag .

# Run
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🎨 UI Access

```bash
# Start Gradio UI
python ui.py

# Visit: http://localhost:7860
```

## 🔧 Configuration Options

**In rag_engine.py:**
```python
# Use different model
rag = RAGEngine(
    vector_store,
    model_name="gpt-4",  # or "gpt-3.5-turbo"
    temperature=0.0
)

# Adjust retrieval
result = rag.query(
    question="...",
    k=10,  # More sources
    use_history=True  # Use conversation context
)
```

**In vector_store.py:**
```python
# Use free local embeddings
vector_store = VectorStore(
    embedding_model="huggingface"  # No API key needed
)
```

## 📊 File Support

- ✅ PDF (`.pdf`)
- ✅ Word (`.docx`)
- ✅ Text (`.txt`)
- ✅ URLs (`http://...`)

## 🚨 Troubleshooting

**No API key?**
```bash
export OPENAI_API_KEY='sk-...'
# Or add to .env file
```

**ChromaDB errors?**
```bash
# Clear and restart
rm -rf vector_db/
python rag_engine.py document.pdf
```

**Slow queries?**
```python
# Use fewer sources
result = rag.query("...", k=2)

# Or use gpt-3.5-turbo (faster)
```

## 📈 Performance Tips

1. **Chunk size**: 1000 tokens (default) works well
2. **Retrieval**: k=4 balances speed/quality
3. **Model**: gpt-3.5-turbo = fast, gpt-4 = better quality
4. **Embeddings**: OpenAI = better, HuggingFace = free

## 🎯 Common Commands

```bash
# Process single document
python document_processor.py doc.pdf

# Test vector search
python vector_store.py doc.pdf

# Interactive Q&A
python rag_engine.py doc.pdf

# Start API
python app.py

# Run UI
python ui.py

# Run tests
pytest test_rag.py
```

## 📝 Project Structure

```
docchat-rag/
├── document_processor.py  # Load & chunk docs
├── vector_store.py        # ChromaDB interface
├── rag_engine.py         # RAG logic
├── app.py                # FastAPI
├── ui.py                 # Gradio UI
├── test_rag.py          # Tests
├── requirements.txt      # Dependencies
└── .env                 # Config (create from .env.example)
```

## 🎓 Learn More

- **RAG**: Retrieval Augmented Generation
- **Embeddings**: Vector representations of text
- **Vector DB**: Semantic search database
- **LLM**: Large Language Model (GPT)
- **Chunking**: Splitting docs into pieces

## ✨ Tips

- Upload multiple docs before querying
- Be specific with questions
- Check sources to verify answers
- Use conversation history for follow-ups
- Try different models for quality/speed trade-off

---

**Built with:** LangChain • ChromaDB • OpenAI • FastAPI • Gradio
