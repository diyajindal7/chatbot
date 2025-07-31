# Langchain RAG Tutorial - FREE Version

This is a free version of the RAG tutorial that uses Hugging Face models instead of OpenAI.

## 🆓 Free Features

- **Embeddings**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) - runs locally, no API costs
- **Text Generation**: Uses Hugging Face Inference API - free tier available
- **Vector Database**: ChromaDB - completely free and local

## Install dependencies

1. Install the free requirements:
```bash
pip install -r requirements_free.txt
```

2. Install markdown dependencies:
```bash
pip install "unstructured[md]"
```

## Setup (Optional)

For text generation, you can optionally get a free Hugging Face API key:

1. Go to https://huggingface.co/settings/tokens
2. Create a free account
3. Generate a new token
4. Add it to your `.env` file:
```
HUGGINGFACE_API_KEY=your_token_here
```

**Note**: The embeddings work completely offline without any API key!

## Create database

Create the Chroma DB using free embeddings:

```bash
python create_database.py
```

## Query the database

Query the Chroma DB:

```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

## How it works

1. **Embeddings**: Uses Sentence Transformers locally - no API calls needed
2. **Vector Search**: ChromaDB finds similar text chunks
3. **Text Generation**: Uses Hugging Face's free inference API (optional)

## Benefits of this version

- ✅ **Completely free** - no OpenAI costs
- ✅ **Works offline** for embeddings
- ✅ **Fast** - local embedding generation
- ✅ **Privacy** - your data stays local
- ✅ **Reliable** - no API rate limits for embeddings

## Troubleshooting

If you don't have a Hugging Face API key, the system will still work for finding relevant text chunks, but won't generate AI responses. You'll see the relevant context instead.

## Model Details

- **Embeddings**: `all-MiniLM-L6-v2` - 384-dimensional embeddings, very fast
- **Text Generation**: `google/flan-t5-base` - good for question answering
- **Vector Database**: ChromaDB - persistent, local storage 