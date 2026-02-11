from typing import Any
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

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
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                splits = text_splitter.split_documents(documents)
                
                # Create embeddings and vector store
                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                
                st.session_state.vectorstore = vectorstore
                st.success(f"✅ Processed {len(splits)} chunks!")
                
                # Clean up temp file
                os.unlink(tmp_path)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This RAG chatbot uses:
    - **LangChain** for orchestration
    - **OpenAI GPT-4.1-mini** for answers
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
                # Create QA chain
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                # Initialize memory if not exists
                if 'memory' not in st.session_state:
                    st.session_state.memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer"
                    )

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=st.session_state.memory,
                    return_source_documents=True
                )

                # Get response
                result = qa_chain.invoke({"question": prompt})
                answer = result['answer']
                
                st.markdown(answer)
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(result['source_documents']):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

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