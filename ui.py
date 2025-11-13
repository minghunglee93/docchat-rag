"""
Gradio UI for DocChat RAG System
Simple web interface for document Q&A
"""

import gradio as gr
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_engine import RAGEngine
from vector_store import VectorStore

# Load environment
load_dotenv()

# Initialize RAG system
print("Initializing DocChat RAG system...")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠ Warning: OPENAI_API_KEY not found. Using HuggingFace embeddings.")
    embedding_model = "huggingface"
else:
    embedding_model = "openai"

# Initialize
vector_store = VectorStore(
    collection_name="docchat_ui",
    embedding_model=embedding_model
)

rag = RAGEngine(
    vector_store=vector_store,
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

print("✓ System ready!")


def upload_file(file):
    """Handle file upload"""
    if file is None:
        return "⚠ No file selected"

    try:
        # Get file path
        file_path = file.name

        # Process and add
        result = rag.add_documents([file_path])

        if result["status"] == "success":
            return f"✓ {result['message']}"
        else:
            return f"✗ Error: {result['message']}"

    except Exception as e:
        return f"✗ Error: {str(e)}"


def add_url(url):
    """Handle URL input"""
    if not url or not url.strip():
        return "⚠ Please enter a URL"

    try:
        result = rag.add_documents([url.strip()])

        if result["status"] == "success":
            return f"✓ {result['message']}"
        else:
            return f"✗ Error: {result['message']}"

    except Exception as e:
        return f"✗ Error: {str(e)}"


def query_documents(question, num_sources, use_history):
    """Query the RAG system"""
    if not question or not question.strip():
        return "⚠ Please enter a question", ""

    try:
        # Check if we have documents
        if vector_store.get_document_count() == 0:
            return "⚠ No documents loaded. Please upload documents first.", ""

        # Query
        result = rag.query(
            question=question.strip(),
            k=num_sources,
            include_sources=True,
            use_history=use_history
        )

        answer = result["answer"]

        # Format sources
        sources_text = ""
        if "sources" in result and result["sources"]:
            sources_text = "\n\n📚 **Sources:**\n\n"
            for i, source in enumerate(result["sources"], 1):
                score = source["relevance_score"]
                src = source["metadata"].get("source", "Unknown")
                content = source["content"][:200] + "..."

                sources_text += f"**{i}.** Score: {score:.3f}\n"
                sources_text += f"Source: `{Path(src).name}`\n"
                if "page" in source["metadata"]:
                    sources_text += f"Page: {source['metadata']['page']}\n"
                sources_text += f"*{content}*\n\n"

        return answer, sources_text

    except Exception as e:
        return f"✗ Error: {str(e)}", ""


def get_system_info():
    """Get system information"""
    try:
        info = rag.get_system_info()
        stats = vector_store.get_statistics()

        info_text = f"""
### System Information

**Model:** {info['model']}
**Total Documents:** {stats['total_documents']}
**Unique Sources:** {stats['unique_sources']}
**Conversation Turns:** {info['conversation_turns']}

**Loaded Sources:**
"""
        for source in stats.get('sources', []):
            info_text += f"\n- `{Path(source).name}`"

        return info_text

    except Exception as e:
        return f"Error: {str(e)}"


def clear_chat_history():
    """Clear conversation history"""
    rag.clear_history()
    return "✓ History cleared!"


def get_chat_history():
    """Get conversation history"""
    history = rag.get_history()

    if not history:
        return "No conversation history yet."

    history_text = "### Conversation History\n\n"
    for i, (q, a) in enumerate(history, 1):
        history_text += f"**{i}. Q:** {q}\n\n"
        history_text += f"**A:** {a}\n\n"
        history_text += "---\n\n"

    return history_text


# Create Gradio interface
with gr.Blocks(title="DocChat - AI Document Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 DocChat - AI Document Q&A

    Upload documents and ask questions. Powered by RAG (Retrieval Augmented Generation).
    """)

    with gr.Tabs():
        # Tab 1: Upload Documents
        with gr.Tab("📄 Upload Documents"):
            gr.Markdown("### Upload files or add URLs")

            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".docx", ".txt"]
                    )
                    upload_btn = gr.Button("Upload File", variant="primary")
                    upload_status = gr.Textbox(label="Status", lines=2)

                with gr.Column():
                    url_input = gr.Textbox(
                        label="Or enter URL",
                        placeholder="https://example.com/document"
                    )
                    url_btn = gr.Button("Add URL", variant="primary")
                    url_status = gr.Textbox(label="Status", lines=2)

            upload_btn.click(upload_file, inputs=file_upload, outputs=upload_status)
            url_btn.click(add_url, inputs=url_input, outputs=url_status)

        # Tab 2: Ask Questions
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Query your documents")

            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What is this document about?",
                        lines=2
                    )

                    with gr.Row():
                        num_sources = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of sources to retrieve"
                        )
                        use_history = gr.Checkbox(
                            label="Use conversation history",
                            value=False
                        )

                    query_btn = gr.Button("Ask Question", variant="primary")

                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=6
                    )

                    sources_output = gr.Markdown(label="Sources")

            query_btn.click(
                query_documents,
                inputs=[question_input, num_sources, use_history],
                outputs=[answer_output, sources_output]
            )

            # Example questions
            gr.Markdown("### Example Questions")
            gr.Examples(
                examples=[
                    ["What is this document about?"],
                    ["Summarize the key points"],
                    ["What are the main topics discussed?"],
                    ["List the important conclusions"]
                ],
                inputs=question_input
            )

        # Tab 3: History
        with gr.Tab("📜 History"):
            gr.Markdown("### Conversation History")

            with gr.Row():
                view_history_btn = gr.Button("View History")
                clear_history_btn = gr.Button("Clear History", variant="stop")

            history_output = gr.Markdown()
            clear_status = gr.Textbox(label="Status", lines=1)

            view_history_btn.click(get_chat_history, outputs=history_output)
            clear_history_btn.click(clear_chat_history, outputs=clear_status)

        # Tab 4: System Info
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("### System Information")

            info_btn = gr.Button("Refresh Info")
            info_output = gr.Markdown()

            info_btn.click(get_system_info, outputs=info_output)

            # Load info on startup
            demo.load(get_system_info, outputs=info_output)

    gr.Markdown("""
    ---
    **Tips:**
    - Upload documents first before asking questions
    - Use specific questions for better answers
    - Check sources to verify information
    - Enable conversation history for follow-up questions
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )