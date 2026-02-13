from langchain_core.messages.tool import ToolMessage
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import ToolMessage

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'checkpointer' not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

MODEL_NAME = os.getenv('MODEL', 'gpt-4.1-mini')


# Page config
st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖")
st.title("🤖 RAG Document Assistant")
st.markdown("Upload a document and ask questions about it!")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a document to create a knowledge base"
    )
    
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        
        # Process document button
        if st.button("🔄 Process Document"):
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load document
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                
                documents = loader.load()
                doc_name = uploaded_file.name
                for doc in documents:
                    doc.metadata["source_doc"] = doc_name

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                splits = text_splitter.split_documents(documents)
                
                # Create embeddings and vector store
                embeddings = OpenAIEmbeddings()
                # Load existing store or create new one
                if os.path.exists("./chroma_db"):
                    vectorstore = Chroma(
                        persist_directory="./chroma_db",
                        embedding_function=embeddings,
                    )
                    vectorstore.add_documents(splits)
                else:
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory="./chroma_db",
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.active_doc = doc_name
                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k": 4,
                        "filter": {"source_doc": doc_name},
                    }
                )
                retriever_tool = create_retriever_tool(
                    retriever,
                    name="document_search",
                    description="Search the document for relevant information"
                )
                st.session_state.agent = create_agent(
                    model= MODEL_NAME,
                    tools=[retriever_tool],
                    checkpointer=st.session_state.checkpointer,
                    system_prompt=('You are a helpful document assistant. '
                        'Use the doc_search tool to find relevant information from the uploaded document before answering. '
                        'Always cite which parts of the document you used. '
                        'If the document doesn\'t contain the answer, say so.'
                    )
                )
                st.success(f"✅ Processed {len(splits)} chunks!")
                
                # Clean up temp file
                os.unlink(tmp_path)
    
    if st.session_state.vectorstore:
        # Get all unique document names from chroma
        all_docs = list(set(
            m['source_doc'] for m in st.session_state.vectorstore.get()['metadatas']
            if "source_doc" in m
        ))
        if len(all_docs) > 1:
            selected = st.selectbox("📄 Switch document",all_docs,index=all_docs.index(st.session_state.active_doc))
            if selected != st.session_state.active_doc:
                st.session_state.active_doc = selected
                # Rebuild retriever with new filter
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={
                        "k": 4,
                        "filter": {"source_doc": selected},
                    }
                )
                retriever_tool = create_retriever_tool(
                    retriever,
                    name="document_search",
                    description="Search the document for relevant information"
                )
                st.session_state.agent = create_agent(
                    model= MODEL_NAME,
                    tools=[retriever_tool],
                    checkpointer=st.session_state.checkpointer,
                    system_prompt=(
                        "You are a helpful document assistant. "
                        "Use the document_search tool to find relevant information from the uploaded document before answering. "
                        "Always cite which parts of the document you used. "
                        "If the document doesn't contain the answer, say so."
                    ),
                )
                st.success(f"✅ Switched to {selected}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(f"""
    This RAG chatbot uses:
    - **LangChain** for orchestration
    - **OpenAI {MODEL_NAME}** for answers
    - **Chroma** for vector storage
    - **Embeddings** for semantic search
    """)

# Main chat interface
if st.session_state.vectorstore:
    st.markdown("### 💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                config = {'configurable': {'thread_id': 'streamlit-session'}}
                response_container = st.empty()
                full_answer = ""
                for msg,metadata in st.session_state.agent.stream({
                    'messages': [
                        {'role': 'user', 'content': prompt}
                    ]
                }, config=config, stream_mode="messages"):
                    # Only stream AI text tokens (skip tool calls, tool results, etc.)
                    if hasattr(msg, 'content') and msg.content and metadata.get("langgraph_node") == "model":
                        full_answer += msg.content
                        response_container.markdown(full_answer + "▌")
                
                # Final render without cursor
                response_container.markdown(full_answer)
            
            # Save to history ONCE after streaming is done
            st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
else:
    st.info("👈 Upload a document from the sidebar to get started!")
    
    # Example section
    st.markdown("### 🎯 What can you do?")
    st.markdown("""
    1. Upload a PDF or text document
    2. Click "Process Document" to create the knowledge base
    3. Ask questions about the content
    4. Get AI-powered answers with sources!
    
    **Example questions:**
    - "What are the main topics in this document?"
    - "Summarize the key points"
    - "What does the author say about [topic]?"
    """)