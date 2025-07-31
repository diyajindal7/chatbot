#!/usr/bin/env python3
"""
Test script for the FREE RAG system
This demonstrates how the system works with Hugging Face models
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

def test_rag_system():
    print("🤖 FREE RAG System Test")
    print("=" * 50)
    
    # Test 1: Embeddings (works offline)
    print("\n1️⃣ Testing Embeddings (Local - No API key needed)")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Test queries
    test_queries = [
        "Alice rabbit",
        "Mad Hatter tea party", 
        "Cheshire Cat",
        "Queen of Hearts"
    ]
    
    print("\n📚 Testing Similarity Search:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = db.similarity_search_with_relevance_scores(query, k=2)
        
        if results:
            print(f"✅ Found {len(results)} relevant chunks")
            for i, (doc, score) in enumerate(results[:2]):
                print(f"   Chunk {i+1} (score: {score:.3f}):")
                print(f"   {doc.page_content[:100]}...")
        else:
            print("❌ No relevant chunks found")
    
    # Test 2: Text Generation (optional - needs API key)
    print("\n\n2️⃣ Testing Text Generation (Optional - Needs API key)")
    hf_token = os.environ.get('HUGGINGFACE_API_KEY')
    
    if hf_token:
        print("✅ Hugging Face API key found!")
        try:
            model = HuggingFaceHub(
                repo_id="google/flan-t5-base",
                huggingfacehub_api_token=hf_token,
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )
            
            # Test a simple question
            test_prompt = "What is Alice's Adventures in Wonderland about?"
            response = model.predict(test_prompt)
            print(f"✅ AI Response: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Error with text generation: {e}")
    else:
        print("❌ No Hugging Face API key found")
        print("💡 To get a FREE API key:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a free account")
        print("   3. Generate a new token")
        print("   4. Add to your .env file: HUGGINGFACE_API_KEY=your_token_here")
    
    print("\n" + "=" * 50)
    print("🎉 FREE RAG System Test Complete!")
    print("\nKey Benefits:")
    print("✅ Embeddings work completely offline")
    print("✅ No OpenAI costs")
    print("✅ Fast local processing")
    print("✅ Privacy - your data stays local")

if __name__ == "__main__":
    test_rag_system() 