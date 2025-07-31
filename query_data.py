import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB with Hugging Face embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:  # Lowered threshold from 0.7 to 0.3
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")

    # Use Hugging Face Hub for text generation (free tier available)
    # You can get a free API key from https://huggingface.co/settings/tokens
    hf_token = os.environ.get('HUGGINGFACE_API_KEY')
    
    if hf_token:
        # Use Hugging Face Inference API (free tier)
        try:
            model = HuggingFaceEndpoint(
                endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
                huggingfacehub_api_token=hf_token,
                task="text2text-generation",
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )
            response_text = model.predict(prompt)
        except Exception as e:
            print(f"Error with text generation: {e}")
            response_text = "Text generation failed, but similarity search worked perfectly!"
    else:
        # Fallback: just show the relevant context
        print("No Hugging Face API key found. Showing relevant context only:")
        response_text = "Please set HUGGINGFACE_API_KEY in your .env file for text generation."

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
