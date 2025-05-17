import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import sys
from typing import List, Dict

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("Error: OPENAI_API_KEY not found")
    sys.exit(1)

# Initialize clients
try:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key, model_name="text-embedding-3-small"
    )
    chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
    collection = chroma_client.get_or_create_collection(
        name="document_qa_collection", embedding_function=openai_ef
    )
    client = OpenAI(api_key=openai_key)
except Exception as e:
    print(f"Error initializing clients: {str(e)}")
    sys.exit(1)

def load_documents_from_directory(directory_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return []
    
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                    documents.append({"id": filename, "text": file.read()})
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
    return documents

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def get_openai_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return []

def query_documents(question: str, n_results: int = 2) -> List[str]:
    try:
        results = collection.query(query_texts=[question], n_results=n_results)
        return [doc for sublist in results["documents"] for doc in sublist]
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        return []

def generate_response(question: str, relevant_chunks: List[str]) -> str:
    if not relevant_chunks:
        return "No relevant information found."
    
    try:
        context = "\n\n".join(relevant_chunks)
        prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of "
            "retrieved context to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Error generating response."

def main():
    # Load and process documents
    documents = load_documents_from_directory("./news_articles")
    if not documents:
        return

    # Split documents into chunks
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    # Generate embeddings and upsert to Chroma
    for doc in chunked_documents:
        embedding = get_openai_embedding(doc["text"])
        if embedding:
            try:
                collection.upsert(
                    ids=[doc["id"]], 
                    documents=[doc["text"]], 
                    embeddings=[embedding]
                )
            except Exception as e:
                print(f"Error upserting chunk {doc['id']}: {str(e)}")

    # Example query
    #question = "What is the maximum estimated cost (in millions of dollars) that Coinbase might spend to fix the incident?"
    question = "How much ransom money did the hackers demand from Coinbase?"
    relevant_chunks = query_documents(question)
    if relevant_chunks:
        answer = generate_response(question, relevant_chunks)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()