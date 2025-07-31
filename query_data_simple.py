#!/usr/bin/env python3
"""
Simple RAG Query Tool - FREE Version
Focuses on the working parts: embeddings and similarity search
"""

import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    print(f"🔍 Searching for: '{query_text}'")
    print("=" * 50)

    # Prepare the DB with Hugging Face embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0:
        print("❌ No relevant results found.")
        return

    print(f"✅ Found {len(results)} relevant chunks:")
    print()

    # Display results with scores
    for i, (doc, score) in enumerate(results, 1):
        print(f"📄 Chunk {i} (Relevance Score: {score:.3f}):")
        print("-" * 40)
        print(doc.page_content)
        print()
        print(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 50)
        print()

    # Summary
    print("🎯 Summary:")
    print(f"• Query: '{query_text}'")
    print(f"• Found {len(results)} relevant text chunks")
    print(f"• Best match score: {results[0][1]:.3f}")
    print("• All processing done locally - $0 cost! 🆓")

if __name__ == "__main__":
    main() 