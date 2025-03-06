import chromadb
from chromadb.utils import embedding_functions
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os
from mistral_embedding_function import MistralEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import time

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def fetch_arxiv_papers(search_term=None, max_results=100):
    # Compute the date range for the past week
    end_date = datetime.utcnow().strftime("%Y%m%d")
    start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y%m%d")
    
    # Build the query URL
    base_url = "http://export.arxiv.org/api/query?"
    if search_term:
        # URL encode the search term
        encoded_search = requests.utils.quote(search_term)
        query = f"search_query=all:{encoded_search}+AND+lastUpdatedDate:[{start_date} TO {end_date}]"
    else:
        query = f"search_query=lastUpdatedDate:[{start_date} TO {end_date}]"
    
    url = f"{base_url}{query}&max_results={max_results}"
    
    # Make the API request
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the XML response
    root = ET.fromstring(response.text)
    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    for entry in root.findall('arxiv:entry', namespace):
        title = entry.find('arxiv:title', namespace).text.strip()
        abstract = entry.find('arxiv:summary', namespace).text.strip()
        # Get PDF URL from arxiv link
        pdf_url = None
        for link in entry.findall('arxiv:link', namespace):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        papers.append({
            "title": title,
            "abstract": abstract,
            "pdf_url": pdf_url
        })
    
    return papers

def split_paper_into_chunks(paper: Dict) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
    )
    
    # Combine title, abstract, and full text
    full_content = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
    if paper.get('full_text'):
        full_content += f"\nFull Text: {paper['full_text']}"
    
    # Split the text into chunks
    chunks = text_splitter.split_text(full_content)
    
    # Create a list of paper chunks with metadata
    paper_chunks = []
    for i, chunk in enumerate(chunks):
        paper_chunks.append({
            "title": paper["title"],
            "content": chunk,
            "chunk_id": i,
            "pdf_url": paper.get("pdf_url"),
            "total_chunks": len(chunks)
        })
    
    return paper_chunks

def import_papers_to_chroma(papers: List[Dict], collection_name: str = "arxiv_papers"):
    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    embedding_function = MistralEmbeddingFunction(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model_name="mistral-embed"
    )
    
    # Try to get existing collection or create new one
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"Found existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"Created new collection: {collection_name}")
    
    # Process each paper and split into chunks
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    for paper_idx, paper in enumerate(papers):
        paper_chunks = split_paper_into_chunks(paper)
        
        for chunk in paper_chunks:
            all_documents.append(chunk["content"])
            all_metadatas.append({
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "total_chunks": chunk["total_chunks"],
                "pdf_url": chunk["pdf_url"],
                "type": "research_paper"
            })
            all_ids.append(f"paper_{paper_idx}_chunk_{chunk['chunk_id']}")
    
    # Add documents in smaller batches with rate limiting
    batch_size = 8  # Reduced batch size
    for i in range(0, len(all_documents), batch_size):
        end_idx = min(i + batch_size, len(all_documents))
        print(f"Processing batch {i//batch_size + 1}, documents {i} to {end_idx}")
        
        try:
            collection.add(
                documents=all_documents[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            # Add a delay between batches (2 seconds)
            time.sleep(0.5)
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(2)  # Wait longer if we hit the rate limit
                # Retry the same batch
                collection.add(
                    documents=all_documents[i:end_idx],
                    metadatas=all_metadatas[i:end_idx],
                    ids=all_ids[i:end_idx]
                )
            else:
                raise e
    
    return collection

def main():
    # Example usage
    # Fetch papers with an optional search term
    papers = fetch_arxiv_papers(search_term="machine learning", max_results=10)
    
    # Import to ChromaDB
    collection = import_papers_to_chroma(papers)
    print(f"Imported {len(papers)} papers to ChromaDB")

if __name__ == "__main__":
    main()