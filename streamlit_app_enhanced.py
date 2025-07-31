#!/usr/bin/env python3
"""
ENHANCED FREE RAG Streamlit App
A user-friendly interface for the FREE RAG system with optional LLM integration
"""

import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import time
from dotenv import load_dotenv

# Import our RAG components
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf']

# Page config
st.set_page_config(
    page_title="Enhanced FREE RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'db_created' not in st.session_state:
        st.session_state.db_created = False
    if 'use_llm' not in st.session_state:
        st.session_state.use_llm = False

def load_documents(file_paths):
    """Load documents from uploaded files with proper error handling"""
    documents = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension in ['.txt', '.md']:
                # Use TextLoader for text files
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Loaded: {Path(file_path).name}")
                
            elif file_extension == '.pdf':
                # Use PyPDFLoader for PDF files
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Loaded PDF: {Path(file_path).name}")
                
            else:
                failed_files.append(f"Unsupported file type: {Path(file_path).name}")
                
        except Exception as e:
            failed_files.append(f"{Path(file_path).name}: {str(e)}")
    
    # Report failed files
    if failed_files:
        st.error("❌ Failed to load some files:")
        for failed in failed_files:
            st.error(f"   - {failed}")
    
    return documents if documents else None

def split_text(documents):
    """Split documents into chunks with better parameters and metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Increased from 300 to 600 for better context
        chunk_overlap=150,  # Increased overlap for better continuity
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for cleaner splits
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add additional metadata to each chunk
    for chunk in chunks:
        # Extract file name from source path
        if 'source' in chunk.metadata:
            file_path = chunk.metadata['source']
            file_name = Path(file_path).name
            chunk.metadata['file_name'] = file_name
            chunk.metadata['file_type'] = Path(file_path).suffix.lower()
        
        # Add chunk index for better tracking
        if 'start_index' in chunk.metadata:
            chunk.metadata['chunk_id'] = f"{chunk.metadata.get('file_name', 'unknown')}_{chunk.metadata['start_index']}"
    
    return chunks

def create_database(chunks):
    """Create the vector database with proper cleanup"""
    try:
        # Clear out the database first with retry logic
        if os.path.exists(CHROMA_PATH):
            try:
                shutil.rmtree(CHROMA_PATH)
            except PermissionError:
                st.warning("⚠️ Database is in use. Trying to force cleanup...")
                import time
                time.sleep(2)  # Wait a moment
                try:
                    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
                except Exception as e:
                    st.error(f"❌ Cannot remove existing database: {str(e)}")
                    st.info("💡 Try refreshing the page or restarting the app")
                    return False, "Database locked by another process"
        
        # Create embeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create database with error handling
        try:
            db = Chroma.from_documents(
                chunks, embedding_function, persist_directory=CHROMA_PATH
            )
            # Remove the persist() call as it's deprecated
            return True, len(chunks)
        except Exception as e:
            st.error(f"❌ Database creation failed: {str(e)}")
            return False, str(e)
            
    except Exception as e:
        return False, str(e)

def query_database(question, k=5):
    """Query the database with MMR for more relevant and diverse results"""
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Use MMR (Max Marginal Relevance) for better diversity
        results = db.max_marginal_relevance_search(
            question, 
            k=k, 
            fetch_k=10,  # Fetch more candidates for better selection
            lambda_mult=0.7  # Balance between relevance and diversity
        )
        
        # Get relevance scores for the MMR results
        relevance_results = db.similarity_search_with_relevance_scores(question, k=k)
        
        # Combine MMR results with relevance scores
        combined_results = []
        for i, doc in enumerate(results):
            if i < len(relevance_results):
                score = relevance_results[i][1]
            else:
                score = 0.5  # Default score if not available
            combined_results.append((doc, score))
        
        # Filter out negative scores and handle empty results
        filtered_results = [(doc, score) for doc, score in combined_results if score > 0]
        
        if not filtered_results:
            return False, "No relevant results found"
        
        return True, filtered_results
    except Exception as e:
        return False, str(e)

def create_improved_prompt(context_text, question):
    """Create an improved prompt template for better responses"""
    prompt = f"""You are a helpful assistant. Use ONLY the following context to answer the user's question. Be specific, quote directly if needed, and do not make up anything. If the answer isn't in the context, say 'I don't know.'

Context: {context_text}

Question: {question}

Answer:"""
    return prompt

def get_llm_response(prompt):
    """Get response from Hugging Face LLM (optional)"""
    try:
        hf_token = os.environ.get('HUGGINGFACE_API_KEY')
        if not hf_token:
            return None, "No Hugging Face API key found"
        
        # Use a good free model for text generation
        model = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
            huggingfacehub_api_token=hf_token,
            task="text2text-generation",
            model_kwargs={"temperature": 0.3, "max_length": 512}
        )
        
        response = model.predict(prompt)
        return True, response
    except Exception as e:
        return False, str(e)

def format_response_with_sources(results, question, use_llm=False):
    """Format response with improved structure and source tracking"""
    if not results:
        return "I don't know. The provided context doesn't contain information to answer this question."
    
    # Create improved prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = create_improved_prompt(context_text, question)
    
    # Format the response with source information
    response_parts = []
    response_parts.append("🔍 **Answer:**\n")
    
    if use_llm:
        # Try to get LLM response
        success, llm_response = get_llm_response(prompt)
        if success:
            response_parts.append(f"{llm_response}\n\n")
        else:
            response_parts.append("Based on the provided context, here are the most relevant sections:\n\n")
    else:
        response_parts.append("Based on the provided context, here are the most relevant sections:\n\n")
    
    # Show sources
    response_parts.append("📚 **Sources:**\n")
    for i, (doc, score) in enumerate(results, 1):
        response_parts.append(f"**Source {i}** (Relevance: {score:.3f}):\n")
        response_parts.append(f"📄 File: {doc.metadata.get('file_name', 'Unknown')}\n")
        if 'page' in doc.metadata:
            response_parts.append(f"📑 Page: {doc.metadata['page'] + 1}\n")
        response_parts.append(f"📝 Content:\n{doc.page_content}\n")
        response_parts.append("---\n")
    
    response_parts.append(f"\n💡 **Summary:** Found {len(results)} relevant sections from your documents.")
    response_parts.append(f"Best match relevance: {results[0][1]:.3f}")
    if use_llm:
        response_parts.append("\n🤖 **Enhanced with AI generation**")
    response_parts.append("\n🆓 **All processing done locally - $0 cost!**")
    
    return "".join(response_parts)

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return temp_dir, file_paths

def check_database_status():
    """Check if database exists and is accessible"""
    if not os.path.exists(CHROMA_PATH):
        return False, "Database does not exist"
    
    try:
        # Try to access the database
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        # Try a simple query to test access
        _ = db.similarity_search("test", k=1)
        return True, "Database is ready"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def force_cleanup_database():
    """Force cleanup of the database directory"""
    try:
        if os.path.exists(CHROMA_PATH):
            import time
            time.sleep(1)  # Brief pause
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            time.sleep(1)  # Another pause
        return True
    except Exception:
        return False

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced FREE RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Hugging Face Models - 100% Free! 🆓")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            help="Upload text files (TXT, MD) or PDF files to create your knowledge base"
        )
        
        # Show uploaded files
        if uploaded_files:
            st.subheader("📁 Uploaded Files:")
            for file in uploaded_files:
                file_extension = Path(file.name).suffix.lower()
                if file_extension in ['.txt', '.md']:
                    st.info(f"📄 {file.name} (Text file)")
                elif file_extension == '.pdf':
                    st.info(f"📕 {file.name} (PDF file)")
                else:
                    st.warning(f"⚠️ {file.name} (Unsupported)")
        
        # Enhanced settings
        st.subheader("🔧 Settings")
        
        # LLM toggle
        use_llm = st.checkbox(
            "🤖 Enable AI Generation (Optional)",
            value=st.session_state.use_llm,
            help="Use Hugging Face LLM for ChatGPT-like responses. Requires API key."
        )
        st.session_state.use_llm = use_llm
        
        if use_llm:
            hf_token = os.environ.get('HUGGINGFACE_API_KEY')
            if not hf_token:
                st.warning("⚠️ No Hugging Face API key found. Add HUGGINGFACE_API_KEY to your .env file for AI generation.")
            else:
                st.success("✅ Hugging Face API key found!")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2 (Recommended)", "sentence-transformers/all-mpnet-base-v2"],
            help="Choose the embedding model for text processing"
        )
        
        # Ingest button
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        # Save uploaded files
                        temp_dir, file_paths = save_uploaded_files(uploaded_files)
                        
                        # Load documents
                        documents = load_documents(file_paths)
                        if documents is None:
                            st.error("❌ No documents could be loaded. Please check your files.")
                            return
                        
                        st.success(f"✅ Successfully loaded {len(documents)} documents")
                        
                        # Split text
                        chunks = split_text(documents)
                        st.success(f"✅ Split into {len(chunks)} chunks")
                        
                        # Create database
                        success, result = create_database(chunks)
                        if success:
                            st.session_state.db_created = True
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Database created with {result} chunks!")
                            st.balloons()
                        else:
                            st.error(f"❌ Failed to create database: {result}")
                        
                        # Cleanup
                        shutil.rmtree(temp_dir)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
        
        # Status
        st.subheader("📊 Status")
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded")
        else:
            st.info("📝 No documents loaded yet")
            
        if st.session_state.db_created:
            st.success("✅ Database ready")
        else:
            st.info("🗄️ Database not created yet")
        
        # Database troubleshooting
        st.subheader("🔧 Database Tools")
        if st.button("🔍 Check Database Status"):
            status, message = check_database_status()
            if status:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
        
        if st.button("🧹 Force Cleanup Database"):
            if force_cleanup_database():
                st.success("✅ Database cleaned up successfully")
                st.session_state.db_created = False
                st.rerun()
            else:
                st.error("❌ Failed to cleanup database")
        
        # Info box
        st.markdown("""
        <div class="info-box">
        <h4>💡 How it works:</h4>
        <ul>
        <li>Upload your documents (TXT, MD, PDF)</li>
        <li>Click "Process Documents" to create embeddings</li>
        <li>Ask questions in the chat below</li>
        <li>Get relevant answers from your documents!</li>
        </ul>
        <h4>📋 Supported files:</h4>
        <ul>
        <li>📄 Text files (.txt, .md)</li>
        <li>📕 PDF files (.pdf)</li>
        </ul>
        <h4>🤖 AI Features:</h4>
        <ul>
        <li>Enable AI generation for ChatGPT-like responses</li>
        <li>Uses MMR for better result diversity</li>
        <li>Improved text splitting for better context</li>
        </ul>
        <h4>🛠️ Troubleshooting:</h4>
        <ul>
        <li>If database is locked, try "Force Cleanup"</li>
        <li>Refresh the page if files won't process</li>
        <li>Check file formats are supported</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main chat area
    st.header("💬 Chat with Your Documents")
    
    if not st.session_state.db_created:
        st.info("👆 Please upload and process documents in the sidebar first!")
        return
    
    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Query database
        with st.spinner("Searching for relevant information..."):
            success, results = query_database(question)
            
            if success and results:
                # Use improved response formatting
                response = format_response_with_sources(results, question, st.session_state.use_llm)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Display response
                with st.chat_message("assistant"):
                    st.write(response)
            else:
                error_msg = "❌ No relevant information found. Try rephrasing your question or uploading more relevant documents."
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                
                with st.chat_message("assistant"):
                    st.write(error_msg)
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main() 