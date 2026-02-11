# RAG Document Assistant 🤖

An intelligent document Q&A system built with LangChain, OpenAI, and Streamlit.

## Features

- 📄 **Document Upload**: Support for PDF and TXT files
- 🔍 **Semantic Search**: Uses vector embeddings for accurate retrieval
- 💬 **Conversational**: Maintains context across questions
- 📚 **Source Citations**: Shows which document sections were used
- ⚡ **Fast**: Optimized chunk sizes and retrieval

## Architecture

```
User Document → Text Splitter → Embeddings → Vector Store (Chroma)
                                                    ↓
User Question → Embeddings → Retrieval → LLM → Answer + Sources
```

## Tech Stack

- **LangChain**: Orchestration framework
- **OpenAI GPT-4.1-mini**: Language model
- **Chroma**: Vector database
- **Streamlit**: Web interface
- **Python**: Backend logic

## Local Setup

```bash
# Clone and install
git clone [your-repo]
cd rag-chatbot
pip install -r requirements.txt

# Add API key
echo "OPENAI_API_KEY=your_key" > .env

# Run
streamlit run app.py
```

## Use Cases

- Research paper Q&A
- Documentation assistant
- Resume/CV analysis
- Legal document review
- Technical manual navigation

## Demo

[Add screenshots here]

## Technical Highlights

- **Chunk Strategy**: 1000 char chunks with 200 char overlap for context preservation
- **Retrieval**: Top-3 most relevant chunks using cosine similarity
- **Memory**: Conversation buffer maintains chat context
- **Embeddings**: text-embedding-ada-002 for semantic understanding

## Future Enhancements

- [ ] Multiple document support
- [ ] Export conversation history
- [ ] Custom chunk size configuration
- [ ] Support for more file types (DOCX, CSV)

## Author

Muhammad Hammad Khalid - Full Stack Developer

- LinkedIn: [https://www.linkedin.com/in/muhammad-hammad-khalid-096488219]
- GitHub: [https://github.com/hammadkh0]
- Email: hammad5718@gmail.com

Built as part of exploring AI/ML integration patterns for production applications.
