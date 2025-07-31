# 🤖 FREE RAG Streamlit App

A beautiful, user-friendly web interface for the FREE RAG system using Hugging Face models.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   pip install "unstructured[md]"
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** and go to `http://localhost:8501`

## ✨ Features

### 📚 Document Upload
- **Multiple file support**: Upload TXT, MD, PDF files
- **Drag & drop interface**: Easy file upload
- **Progress tracking**: See processing status in real-time

### 🔧 Settings Panel
- **Embedding model selection**: Choose different models
- **Processing controls**: Start/stop document ingestion
- **Status indicators**: See database and document status

### 💬 Chat Interface
- **Real-time chat**: Ask questions about your documents
- **Chat history**: View previous conversations
- **Relevance scores**: See how well matches fit your questions
- **Clear chat**: Reset conversation history

### 🎨 Beautiful UI
- **Responsive design**: Works on desktop and mobile
- **Clean layout**: Intuitive navigation
- **Success/error messages**: Clear feedback
- **Loading indicators**: Know when processing is happening

## 🆓 FREE Features

- ✅ **100% Free** - No OpenAI costs
- ✅ **Local Processing** - Embeddings run on your machine
- ✅ **No API Keys** - Works completely offline
- ✅ **Fast** - No network latency for embeddings
- ✅ **Private** - Your data stays local

## 📖 How to Use

### 1. Upload Documents
1. Click "Browse files" in the sidebar
2. Select your text files (TXT, MD, PDF)
3. Click "🚀 Process Documents"
4. Wait for processing to complete

### 2. Ask Questions
1. Type your question in the chat input
2. Press Enter or click send
3. View relevant information from your documents
4. See relevance scores for each chunk

### 3. Explore Results
- **Relevance scores**: Higher scores = better matches
- **Source tracking**: See which documents contain answers
- **Chunk preview**: View the actual text that matched

## 🔧 Technical Details

### Supported File Types
- **TXT**: Plain text files
- **MD**: Markdown files
- **PDF**: PDF documents (basic support)

### Embedding Models
- **all-MiniLM-L6-v2** (Default): Fast, good quality
- **all-mpnet-base-v2**: Higher quality, slower

### Processing Pipeline
1. **Document Loading**: Parse uploaded files
2. **Text Splitting**: Break into 300-character chunks
3. **Embedding Generation**: Convert to vectors using Hugging Face
4. **Vector Storage**: Store in ChromaDB
5. **Similarity Search**: Find relevant chunks for queries

## 🎯 Example Usage

### Upload Documents
```
📁 Upload: alice_in_wonderland.md
📁 Upload: company_manual.txt
📁 Upload: research_paper.pdf
```

### Ask Questions
```
Q: "What happens at the tea party?"
A: [Relevant chunks from Alice in Wonderland]

Q: "What are the company policies?"
A: [Relevant chunks from company manual]

Q: "What are the main findings?"
A: [Relevant chunks from research paper]
```

## 🛠️ Troubleshooting

### Common Issues

**"No documents loaded yet"**
- Upload files using the sidebar
- Click "Process Documents" button

**"Database not created yet"**
- Make sure documents are uploaded
- Click "Process Documents" button
- Wait for processing to complete

**"No relevant information found"**
- Try rephrasing your question
- Upload more relevant documents
- Check if documents contain the information you're looking for

### Performance Tips

- **Smaller files**: Process faster
- **Text files**: Better than PDFs for speed
- **Relevant content**: Upload documents that contain the information you need

## 🎊 Benefits

### Cost Comparison
| Feature | OpenAI Version | FREE Streamlit App |
|---------|----------------|-------------------|
| Embeddings | $0.0001 per 1K tokens | **$0** |
| Text Generation | $0.002 per 1K tokens | **$0** |
| Web Interface | $0 (local) | **$0** |
| **Total** | **~$0.002 per query** | **$0** |

### Privacy & Security
- ✅ **Local Processing**: No data sent to external APIs
- ✅ **No API Keys**: No account creation needed
- ✅ **Offline Capable**: Works without internet
- ✅ **Data Control**: Your documents stay on your machine

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run with specific port
streamlit run streamlit_app.py --server.port 8501

# Run in background
nohup streamlit run streamlit_app.py &
```

## 📝 License

This is a FREE RAG system - no licensing costs!

---

**🎉 Enjoy your FREE RAG chatbot! No costs, no limits, just powerful document Q&A!** 