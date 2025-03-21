import chromadb
from chromadb.utils import embedding_functions
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os
from .mistral_embedding_function import MistralEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import time
from langchain_community.document_loaders import ArxivLoader
import logging

logger = logging.getLogger("arxiv")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


class ArxivAbstractDB:
    """
    A class for managing a vector database of arXiv paper abstracts using ChromaDB.

    This class handles storage and retrieval of arXiv paper abstracts using vector embeddings
    generated by the Mistral API. It provides methods to add new abstracts and query existing ones.

    Attributes:
        db_path (str): Path to the ChromaDB database directory
        chroma_client: ChromaDB client instance
        embedding_function: Mistral embedding function instance
        collection_name (str): Name of the ChromaDB collection
        collection: ChromaDB collection instance
    """

    def __init__(self, db_path: str = "./chroma_abstract_db"):
        """
        Initialize the ArxivAbstractDB.

        Args:
            db_path (str): Path to the ChromaDB database directory
        """
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = MistralEmbeddingFunction(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model_name="mistral-embed"
        )
        self.collection_name = "arxiv_abstracts"
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create new one if it doesn't exist.

        Returns:
            ChromaDB collection instance
        """
        try:
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        return collection

    def add_abstracts(self, abstracts: List[Dict]):
        """
        Add paper abstracts to the database.

        Args:
            abstracts (List[Dict]): List of paper abstracts with metadata to add
        """
        documents = []
        metadatas = []
        ids = []
        
        for paper in abstracts:
            # Create unique ID using title and published date to avoid duplicates
            paper_id = f"abstract_{paper['title']}_{paper.get('published_date', '')}"
            # Remove special characters and spaces to make a valid ID
            paper_id = "".join(c if c.isalnum() else "_" for c in paper_id)
            logger.info(f"paper_id: {paper_id[:30]}")
            
            # Check if document with this ID already exists
            existing_doc = self.collection.get(ids=[paper_id])
            if existing_doc['ids']:
                logger.info(f"Document {paper_id} already exists, skipping...")
                continue
                
            # Combine title and abstract
            content = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            documents.append(content)
            metadatas.append({
                "title": paper["title"],
                "pdf_url": paper.get("pdf_url", ""),
                "authors": paper.get("authors", ""),
                "published_date": paper.get("published_date", ""),
                "categories": paper.get("categories", ""),
                "primary_category": paper.get("primary_category", ""),
                "type": "abstract"
            })
            ids.append(paper_id)
        
        if not documents:  # Skip if no new documents to add
            logger.info("No new abstracts to add")
            return
            
        # Add documents in batches
        batch_size = 16
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            try:
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print("Rate limit hit, waiting 2 seconds...")
                    time.sleep(2)
                    self.collection.add(
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                else:
                    raise e

    def query(self, query_text: str, top_k: int = 5, filter_dict: dict = None) -> List[Dict]:
        """
        Query the abstract database.
        
        Args:
            query_text (str): The query text to search for
            top_k (int): Number of results to return
            filter_dict (dict): Optional dictionary of metadata filters
        
        Returns:
            List[Dict]: List of results with their metadata
        """
        try:
            results = self.collection.query(
                query_texts=query_text,  # Changed from query_texts=[query_text]
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for idx in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][idx],
                    'distance': results['distances'][0][idx],
                    **results['metadatas'][0][idx]  # Unpack metadata
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during query: {e}")
            return []
        
    def check_query_relevance(self, query: str, threshold: float = 0.75) -> tuple[float, bool]: 
        """
        Check if a query is relevant to the content in the vector database.
        
        Args:
            query_text (str): The query text to check
            threshold (float): Similarity threshold (0 to 1) for considering results relevant
            
        Returns:
            tuple[float, bool]: (max_relevance_score, is_relevant)
            - max_relevance_score: Score between 0 and 1 indicating maximum relevance
            - is_relevant: Boolean indicating if query is relevant enough
        """
        try:
            # Query the collection with a single result to get most relevant match
            result = self.query(query, top_k=1)
            
            # Handle empty results
            if not result:
                return 0.0, False
                
            # ChromaDB returns L2 distances, convert to similarity score (0 to 1)
            # Typical L2 distances range from 0 (identical) to 2 (completely different)
            distance = result[0]['distance']
            similarity = 1 - distance # Convert to 0-1 scale

            return similarity, similarity >= threshold
            
        except Exception as e:
            print(f"Error during relevance check: {e}")
            return 0.0, False

class ArxivAbstractFetcher:
    """
    A class for fetching abstracts from arXiv papers.

    This class provides methods to retrieve abstracts and metadata from arXiv papers
    using the arXiv API via LangChain's ArxivLoader.
    """

    def __init__(self):
        """Initialize the ArxivAbstractFetcher."""
        pass
        
    def fetch_arxiv_abstracts(self, search_term=None, max_results=100):
        """
        Fetch abstracts from arXiv based on a search query.

        Args:
            search_term (str, optional): Search query for arXiv. Defaults to AI/ML categories.
            max_results (int, optional): Maximum number of results to fetch. Defaults to 100.

        Returns:
            List[Dict]: List of papers with their abstracts and metadata
        """
        if not search_term:
            search_term = "cat:cs.AI OR cat:cs.LG"  # Default to AI/ML categories if no search term
        
        # Initialize ArxivLoader with the search query
        loader = ArxivLoader(
            query=search_term,
            load_max_docs=max_results,
            load_all_available_meta=True,
            load_full_text=False  # Only load abstracts
        )

        # Load documents (this will fetch PDFs and convert them to text)
        documents = loader.load()
        
        # Convert documents to our paper format
        papers = []
        for doc in documents:
            papers.append({
                "title": doc.metadata.get("Title", ""),
                "abstract": doc.metadata.get("Summary", ""),
                "pdf_url": doc.metadata.get("entry_id", ""),
                "authors": doc.metadata.get("Authors", ""),
                "published_date": doc.metadata.get("Published", ""),
                "categories": ", ".join(doc.metadata.get("categories", [])),
                "primary_category": doc.metadata.get("primary_category", "")
            })
        return papers

class ArxivFullTextDB:
    """
    A class for managing a vector database of full arXiv paper texts using ChromaDB.

    This class handles storage and retrieval of complete arXiv papers using vector embeddings
    generated by the Mistral API. It provides methods to add new papers and query existing ones.

    Attributes:
        db_path (str): Path to the ChromaDB database directory
        chroma_client: ChromaDB client instance
        embedding_function: Mistral embedding function instance
        collection_name (str): Name of the ChromaDB collection
        collection: ChromaDB collection instance
    """

    def __init__(self, db_path: str = "./chroma_fulltext_db"):
        """
        Initialize the ArxivFullTextDB.

        Args:
            db_path (str): Path to the ChromaDB database directory
        """
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = MistralEmbeddingFunction(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model_name="mistral-embed"
        )
        self.collection_name = "arxiv_full_papers"
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create new one if it doesn't exist.

        Returns:
            ChromaDB collection instance
        """
        try:
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Found existing collection: {self.collection_name}")
        except:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new collection: {self.collection_name}")
        return collection
    
    def split_paper_into_chunks(self, paper: Dict) -> List[Dict]:
        """
        Split a paper into smaller chunks for better processing.

        Args:
            paper (Dict): Paper with full text and metadata

        Returns:
            List[Dict]: List of chunks with metadata
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""] # Try to split on paragraph breaks first
        )
        
        # Combine title and abstract as the first chunk
        metadata_chunk = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        
        chunks = []
        
        # Add metadata chunk as the first chunk
        chunks.append({
            "title": paper["title"],
            "content": metadata_chunk,
            "chunk_id": 0,
            "chunk_type": "metadata",
            "pdf_url": paper.get("pdf_url", ""),
            "authors": paper.get("authors", ""),
            "published_date": paper.get("published_date", ""),
            "categories": paper.get("categories", ""),
            "primary_category": paper.get("primary_category", ""),
            "total_chunks": 1  # Will update this after splitting full text
        })
        
        # Split the full text if it exists
        if paper.get('full_text'):
            text_chunks = text_splitter.split_text(paper['full_text'])
            
            # Add each text chunk with metadata
            for i, chunk in enumerate(text_chunks, start=1):
                chunks.append({
                    "title": paper["title"],
                    "content": chunk,
                    "chunk_id": i,
                    "chunk_type": "full_text",
                    "pdf_url": paper.get("pdf_url", ""),
                    "authors": paper.get("authors", ""),
                    "published_date": paper.get("published_date", ""),
                    "categories": paper.get("categories", ""),
                    "primary_category": paper.get("primary_category", ""),
                    "total_chunks": len(text_chunks) + 1  # +1 for metadata chunk
                })
        
        # Update total_chunks in all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks
        
        return chunks

    def add_papers(self, papers: List[Dict]):
        """
        Add full papers to the database.

        Args:
            papers (List[Dict]): List of papers with full text and metadata to add
        """
        for paper in papers:
            # Create unique ID using title and published date to avoid duplicates
            base_paper_id = f"paper_{paper['title']}_{paper.get('published_date', '')}"
            base_paper_id = "".join(c if c.isalnum() else "_" for c in base_paper_id)
            
            # Check if first chunk exists
            first_chunk_id = f"{base_paper_id}_chunk_0"
            existing_doc = self.collection.get(ids=[first_chunk_id])
            if existing_doc['ids']:
                print(f"Paper {base_paper_id} already exists, skipping...")
                continue
            
            # Get chunks for the paper
            chunks = self.split_paper_into_chunks(paper)
            
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Create chunk-specific ID
                chunk_id = f"{base_paper_id}_chunk_{chunk['chunk_id']}"
                
                documents.append(chunk["content"])
                metadatas.append({
                    "title": chunk["title"],
                    "chunk_id": chunk["chunk_id"],
                    "total_chunks": chunk["total_chunks"],
                    "chunk_type": chunk["chunk_type"],
                    "pdf_url": chunk.get("pdf_url", ""),
                    "authors": chunk.get("authors", ""),
                    "published_date": chunk.get("published_date", ""),
                    "categories": chunk.get("categories", ""),
                    "primary_category": chunk.get("primary_category", ""),
                    "type": "full_text"
                })
                ids.append(chunk_id)
            
            if not documents:  # Skip if no chunks to add
                print("No chunks to add for this paper")
                continue
            
            # Add documents in batches
            batch_size = 16
            for i in range(0, len(documents), batch_size):
                print(f"Adding {len(documents[i:i+batch_size])} chunks")
                end_idx = min(i + batch_size, len(documents))
                self.call_embeddings_api(documents[i:end_idx], metadatas[i:end_idx], ids[i:end_idx])

    def call_embeddings_api(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        while True:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                break
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print("Rate limit hit, waiting 2 seconds...")
                    time.sleep(2)
                else:
                    raise e

    def query(self, query_text: str, top_k: int = 5, filter_dict: dict = None) -> List[Dict]:
        """
        Query the full text database.
        
        Args:
            query_text (str): The query text to search for
            top_k (int): Number of results to return
            filter_dict (dict): Optional dictionary of metadata filters
        
        Returns:
            List[Dict]: List of results with their metadata and chunk information
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for idx in range(len(results['documents'][0])):
                print(results['distances'])
                result = {
                    'content': results['documents'][0][idx],
                    'distance': results['distances'][0][idx],
                    **results['metadatas'][0][idx]  # Unpack metadata
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during query: {e}")
            return []
        
    def check_query_relevance(self, query: str, threshold: float = 0.75) -> tuple[float, bool]: 
        """
        Check if a query is relevant to the content in the vector database.
        
        Args:
            query_text (str): The query text to check
            threshold (float): Similarity threshold (0 to 1) for considering results relevant
            
        Returns:
            tuple[float, bool]: (max_relevance_score, is_relevant)
            - max_relevance_score: Score between 0 and 1 indicating maximum relevance
            - is_relevant: Boolean indicating if query is relevant enough
        """
        try:
            # Query the collection with a single result to get most relevant match
            result = self.query(query, top_k=1)
            
            # Handle empty results
            if not result:
                return 0.0, False
                
            # ChromaDB returns L2 distances, convert to similarity score (0 to 1)
            # Typical L2 distances range from 0 (identical) to 2 (completely different)
            distance = result[0]['distance']
            similarity = 1 - distance # Convert to 0-1 scale

            return similarity, similarity >= threshold
            
        except Exception as e:
            print(f"Error during relevance check: {e}")
            return 0.0, False

class ArxivFullTextFetcher:
    """
    A class for fetching full text content from arXiv papers.

    This class provides methods to retrieve the complete text of arXiv papers, including
    the title, abstract, and full PDF content. It can fetch papers either by specific 
    arXiv ID or by search query.

    Methods:
        fetch_arxiv_full_text_from_id: Fetches a single paper's full text by arXiv ID
        fetch_arxiv_full_text_from_query: Fetches multiple papers' full text by search query
    """

    def __init__(self):
        """Initialize the ArxivFullTextFetcher."""
        pass

    def fetch_arxiv_full_text_from_id(self, arxiv_id=None):
        """
        Fetch full text for a single paper by arXiv ID.

        Args:
            arxiv_id (str): arXiv ID of the paper to fetch

        Returns:
            List[Dict]: List containing single paper with full text and metadata
        """
        return self.fetch_arxiv_full_text_from_query(search_term=arxiv_id, max_results=1)

    def fetch_arxiv_full_text_from_query(self, search_term=None, max_results=100):
        """
        Fetch full text for multiple papers based on a search query.

        Args:
            search_term (str, optional): Search query for arXiv. Defaults to AI/ML categories.
            max_results (int, optional): Maximum number of results to fetch. Defaults to 100.

        Returns:
            List[Dict]: List of papers with full text and metadata
        """
        if not search_term:
            search_term = "cat:cs.AI OR cat:cs.LG"  # Default to AI/ML categories if no search term
        
        # Initialize ArxivLoader with the search query
        loader = ArxivLoader(
            query=search_term,
            load_max_docs=max_results,
            load_all_available_meta=True,
            load_full_text=True  # Load full text
        )
        
        # Load documents (this will fetch PDFs and convert them to text)
        documents = loader.load()
        print("Documents:", documents)

        # Convert documents to our paper format
        papers = []
        for doc in documents:
            papers.append({
                "title": doc.metadata.get("Title", ""),
                "abstract": doc.metadata.get("Summary", ""),
                "full_text": doc.page_content,
                "pdf_url": doc.metadata.get("entry_id", ""),
                "authors": doc.metadata.get("Authors", ""),
                "published_date": doc.metadata.get("Published", ""),
                "categories": ", ".join(doc.metadata.get("categories", [])),
                "primary_category": doc.metadata.get("primary_category", "")
            })
        return papers
    


# Check if query in the database
# If not fetch arxiv abstracts


# def main():
#     arxiv_abstract_db = ArxivAbstractDB()
#     arxiv_full_text_db = ArxivFullTextDB()
    
#     # Fetch and add abstracts
#     fetcher = ArxivAbstractFetcher()
#     abstracts = fetcher.fetch_arxiv_abstracts(search_term="transformers", max_results=100)
#     for abstract in abstracts:
#         print(abstract.get("title"))


# if __name__ == "__main__":
#     main()
