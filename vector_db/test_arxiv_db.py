import chromadb
from mistral_embedding_function import MistralEmbeddingFunction
import os

def test_arxiv_collection():
    query_text = "cancer mL"

    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create embedding function
    embedding_function = MistralEmbeddingFunction(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model_name="mistral-embed"
    )
    
    # Get the collection
    collection = chroma_client.get_collection(
        name="arxiv_papers",
        embedding_function=embedding_function
    )
    
    # Test 1: Basic similarity search
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    print(f"\nTest 1 - Similar papers to '{query_text}':")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\nResult {i+1}:")
        print(f"Title: {metadata['title']}")
        print(f"Chunk {metadata['chunk_id']} of {metadata['total_chunks']}")
        print("Content preview:", doc, "...")

    # Test 2: Get all chunks for a specific paper
    paper_title = results['metadatas'][0][0]['title']
    paper_chunks = collection.get(
        where={"title": paper_title}
    )
    print(f"\nTest 2 - All chunks for paper '{paper_title}':")
    print(f"Found {len(paper_chunks['ids'])} chunks")

    # Test 3: Count total unique papers
    all_metadata = collection.get()['metadatas']
    unique_papers = len(set(meta['title'] for meta in all_metadata))
    print(f"\nTest 3 - Total unique papers in collection: {unique_papers}")

if __name__ == "__main__":
    test_arxiv_collection() 