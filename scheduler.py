from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_community.document_loaders import ArxivLoader
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from typing_extensions import Literal
from langchain_core.tools import tool  
from vector_db.arxiv_vector_db import ArxivAbstractDB, ArxivFullTextDB, ArxivAbstractFetcher, ArxivFullTextFetcher
from pydantic import BaseModel, Field
from typing import List, Dict
import time
import json

llm = ChatMistralAI(model="mistral-small-latest")

def call_llm(prompt: str):
    while True:
        try:
            return llm.invoke(prompt)
        except Exception as e:
            time.sleep(1)

def call_structured_llm(structured_llm, prompt: str):
    while True:
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            time.sleep(1)

tasks_workflow = """
The user is asking you to do research on a specific topic. The workflow depends on whether this is the first message in a thread or a follow-up question.

FOR FIRST MESSAGE IN THREAD:
1. Name of Task: Fetch papers from Arxiv API
(Below is information about the task above, do not include it in the task description)
- Inputs: Search term, number of papers to return
- Outputs: List of papers
- Prerequisites: None

2. Name of Task: Store papers in local vector database
(Below is information about the task above, do not include it in the task description)
- Inputs: List of papers
- Outputs: Confirmation of storage
- Prerequisites: 1

3. Name of Task: Initial paper analysis
(Below is information about the task above, do not include it in the task description)
- Inputs: List of papers
- Outputs: Initial analysis based on user's request (summary, authors, key findings, etc.)
- Prerequisites: 2

FOR FOLLOW-UP MESSAGES IN THREAD:
1. Name of Task: Query understanding
(Below is information about the task above, do not include it in the task description)
- Inputs: User query
- Outputs: Structured understanding of what user is asking about the papers
- Prerequisites: None

2. Name of Task: Retrieve relevant content
(Below is information about the task above, do not include it in the task description)
- Inputs: Query understanding, vector database
- Outputs: Relevant sections from stored papers
- Prerequisites: 1

3. Name of Task: Answer user query
(Below is information about the task above, do not include it in the task description)
- Inputs: Retrieved content, query understanding
- Outputs: Targeted response based on user's specific request
- Prerequisites: 2
"""

TASKS_FIRST_MESSAGE = [
    "Fetch papers from Arxiv API",
    "Store papers in local vector database",
    "Initial paper analysis"
]

TASKS_FOLLOW_UP = [
    "Query understanding",
    "Retrieve relevant content",
    "Answer user query"
]

class UserRequest(BaseModel):
    request: str = Field(None, description="Request that is what the user is asking you to do")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="History of previous chat messages")

# Graph state
class ChatbotState(TypedDict):
    search_term: str
    user_request: UserRequest
    new_tasks: List[Dict[str, str]]
    is_new_thread: bool
    fetch_arxiv_papers_info: List[Dict[str, str]]
    store_papers_info: str
    query_vector_db_info: List[Dict[str, str]]
    summarize_abstract_info: List[Dict[str, str]]
    summarize_paper_info: List[Dict[str, str]]
    compress_summaries_info: List[Dict[str, str]]
    final_task_info: List[Dict[str, str]]

class ChatBotTools:
    def __init__(self):
        self.arxiv_abstract_db = ArxivAbstractDB()
        self.arxiv_full_text_db = ArxivFullTextDB()
        self.arxiv_abstract_fetcher = ArxivAbstractFetcher()
        self.arxiv_full_text_fetcher = ArxivFullTextFetcher()

    def extract_search_term_node(self, state: ChatbotState):
        """Extract the search term from the user's request"""
        print("User's request: ", state)
        
        # Different prompts for new thread vs follow-up
        if state.get("is_new_thread", True):
            search_term_prompt = f"""
                User's request: {state["user_request"].request}
                Extract the one search term phrase from the user's request to search Arxiv to get relevant papers.
                Only output the search term, nothing else.
            """
        else:
            search_term_prompt = f"""
                User's request: {state["user_request"].request}
                This is a follow-up question about previously retrieved papers.
                Extract key terms to search within the existing papers.
                Only output the search terms, nothing else.
            """
        
        print("Search term prompt: ", search_term_prompt)
        search_term = call_llm(search_term_prompt)
        print("Search term: ", search_term.content)
        return {"search_term": search_term.content}

    def check_url_in_input_node(self, state: ChatbotState):
        """Check if a URL was provided in the user input and extract it"""
        print("Checking for URL in user input...")
        
        url_extract_prompt = f"""
        User's request: {state["user_request"].request}
        
        Does this request contain a URL (web address) such as an arxiv.org link, DOI, or other research paper identifier?
        Analyze the text carefully for anything that looks like a URL or paper identifier.
        
        If you detect a URL or paper identifier, extract and return ONLY the URL including the https://.
        If no URL or identifier is found, return only "FALSE".
        """
        
        url_extract_result = call_llm(url_extract_prompt)
        result_content = url_extract_result.content.strip()
        
        has_url = result_content.upper() != "FALSE"
        extracted_url = result_content if has_url else ""
        
        print(f"URL detected: {has_url}")
        if has_url:
            print(f"Extracted URL: {extracted_url}")
            # Update the search term to the extracted URL
            return {"search_term": extracted_url}
        
        # If no URL found, keep the original search term
        return {"search_term": state["search_term"]}
    
    def fetch_arxiv_papers_node(self, state: ChatbotState):
        """Fetch papers from Arxiv API based on search term"""
        number_of_papers = 5
        
        # Skip fetching if this is not a new thread
        if not state.get("is_new_thread", True):
            print("Skipping arXiv fetch for follow-up query in existing thread")
            return {"fetch_arxiv_papers_info": []}
        
        if state["new_tasks"][0]["task"] != "None":
            print("Doing task 1: Fetching papers from Arxiv API!")
            
            try:
                # Check if the search term is a URL
                is_url = "arxiv.org" in state["search_term"] or "http" in state["search_term"]
                
                papers = []
                if is_url:
                    # Extract paper ID from URL if it's an arxiv URL
                    paper_id = None
                    if "arxiv.org/abs/" in state["search_term"]:
                        paper_id = state["search_term"].split("arxiv.org/abs/")[1].split()[0]
                    elif "arxiv.org/pdf/" in state["search_term"]:
                        paper_id = state["search_term"].split("arxiv.org/pdf/")[1].split(".pdf")[0]
                    
                    if paper_id:
                        print(f"Fetching specific paper with ID: {paper_id}")
                        try:
                            papers = self.arxiv_full_text_fetcher.fetch_arxiv_full_text_from_id(paper_id)
                        except Exception as e:
                            print(f"Error fetching specific paper: {e}")
                
                # If not a URL or direct fetch failed, use regular query
                if not papers:
                    try:
                        papers = self.arxiv_full_text_fetcher.fetch_arxiv_full_text_from_query(
                            state["search_term"], number_of_papers
                        )
                    except Exception as e:
                        print(f"Error during arxiv query: {e}")
                        # Return a minimal paper structure with error info
                        papers = [{
                            "title": "Error fetching papers",
                            "abstract": f"Failed to fetch papers for query: {state['search_term']}. Please try a different search term.",
                            "full_text": "",
                            "error": str(e)
                        }]
                
                return {"fetch_arxiv_papers_info": papers}
            except Exception as e:
                print(f"Unexpected error in fetch_arxiv_papers_node: {e}")
                return {"fetch_arxiv_papers_info": [{
                    "title": "Error in paper fetch",
                    "abstract": "An unexpected error occurred while fetching papers.",
                    "full_text": "",
                    "error": str(e)
                }]}
        else:
            print("Not doing task 1: Fetch papers from Arxiv API")
            return {"fetch_arxiv_papers_info": []}
    def store_papers_node(self, state: ChatbotState):
        """Store fetched papers in the local vector database"""
        if state["new_tasks"][1]["task"] != "None" and state["fetch_arxiv_papers_info"]:
            print("Doing task 2: Storing papers in local vector database!")
            
            # Store in full text database
            self.arxiv_full_text_db.add_papers(state["fetch_arxiv_papers_info"])
            
            # Also extract abstracts and store in abstract database
            abstracts = []
            for paper in state["fetch_arxiv_papers_info"]:
                abstracts.append({
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "pdf_url": paper.get("pdf_url", ""),
                    "authors": paper.get("authors", ""),
                    "published_date": paper.get("published_date", ""),
                    "categories": paper.get("categories", ""),
                    "primary_category": paper.get("primary_category", "")
                })
            
            self.arxiv_abstract_db.add_abstracts(abstracts)
            
            return {"store_papers_info": f"Successfully stored {len(state['fetch_arxiv_papers_info'])} papers"}
        else:
            print("Not doing task 2: Store papers in local vector database")
            return {"store_papers_info": ""}
    
    def query_vector_db_node(self, state: ChatbotState):
        """Query the local vector database directly with user's query"""
        if state["new_tasks"][1]["task"] != "None":
            print("Querying vector database!")
            
            search_content = state["search_term"]
            print("Search query:", search_content)
            
            try:
                # Query both databases
                abstracts = self.arxiv_abstract_db.query(search_content, 3)
                full_texts = self.arxiv_full_text_db.query(search_content, 2)
                
                # Combine and deduplicate results
                results = full_texts + abstracts
                unique_results = []
                seen_titles = set()
                
                # Limit content length for each result
                for result in results:
                    title = result.get("title", "")
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        # Truncate content if needed
                        if "content" in result:
                            result["content"] = result["content"][:1000] + "..." if len(result["content"]) > 1000 else result["content"]
                        unique_results.append(result)
                
                print(f"Found {len(unique_results)} unique results")
                return {"query_vector_db_info": unique_results[:3]}
                
            except Exception as e:
                print(f"Error during vector DB query: {e}")
                return {"query_vector_db_info": []}
        else:
            return {"query_vector_db_info": []}

    def check_local_vector_database_node(self, state: ChatbotState):
        """Check the local vector database for relevance"""
        number_of_papers = 5
        
        if state["new_tasks"][0]["task"] != "None":
            print("Doing task 1!")

            print("User's request: ", state)
            embellished_search_term = call_llm(
                f"""
                User's request: {state["user_request"].request}
                Search term: {state["search_term"]}
                Given this short search term, write a short abstract (< 200 words) about it so that we can embed the paragraph and search our vector database of abstract s-- In other words, you are embelishing the query for better results in vector search"
                """
            )

            print("Embellished search term: ", embellished_search_term)

            _, is_relevant_abstract = self.arxiv_abstract_db.check_query_relevance(embellished_search_term.content, number_of_papers)
            _, is_relevant_full_text = self.arxiv_full_text_db.check_query_relevance(embellished_search_term.content, number_of_papers)

            print("Is relevant abstract: ", is_relevant_abstract)
            print("Is relevant full text: ", is_relevant_full_text)
            
            if is_relevant_abstract and is_relevant_full_text:
                return {"check_local_vector_database_info": []}
            
            abstracts = []
            if is_relevant_abstract:
                abstracts = self.arxiv_abstract_db.query(state["search_term"], number_of_papers)

            full_texts = []
            if is_relevant_full_text:
                full_texts = self.arxiv_full_text_db.query(state["search_term"], number_of_papers)

            return {"check_local_vector_database_info": abstracts + full_texts}

        else:
            print("Not doing task 1!")
            return {"check_local_vector_database_info": []}
        
    def query_arxiv_api_abstracts_node(self, state: ChatbotState):
        """Check the Arxiv API for relevance"""
        number_of_papers = 5
        
        if state["new_tasks"][1]["task"] != "None" and state["check_local_vector_database_info"] == []:
            print("Doing task 2!")

            abstracts = self.arxiv_abstract_fetcher.fetch_arxiv_abstracts(state["search_term"], number_of_papers)
            print("Abstracts: ", abstracts)
            self.arxiv_abstract_db.add_abstracts(abstracts)

            return {"arxiv_api_abstracts_info": abstracts}

        else:
            print("Not doing task 2!")
            return {"arxiv_api_abstracts_info": []}
        
    def query_arxiv_api_papers_node(self, state: ChatbotState):
        """Check the Arxiv API for relevance"""
        number_of_papers = 5
        
        print("Trying task 3!")
        if state["new_tasks"][2]["task"] != "None" and state["check_local_vector_database_info"] == []:
            print("Doing task 3!")

            # Check if the search term is a URL
            is_url = "arxiv.org" in state["search_term"] or "http" in state["search_term"]
            
            if is_url:
                # Extract paper ID from URL if it's an arxiv URL
                paper_id = None
                if "arxiv.org/abs/" in state["search_term"]:
                    paper_id = state["search_term"].split("arxiv.org/abs/")[1].split()[0]
                elif "arxiv.org/pdf/" in state["search_term"]:
                    paper_id = state["search_term"].split("arxiv.org/pdf/")[1].split(".pdf")[0]
                
                if paper_id:
                    print(f"Fetching specific paper with ID: {paper_id}")
                    # Use the ArxivLoader to fetch the specific paper
                    try:
                        papers = self.arxiv_full_text_fetcher.fetch_arxiv_full_text_from_id(paper_id)
                        self.arxiv_full_text_db.add_papers(papers)
                        return {"arxiv_api_papers_info": papers}
                    except Exception as e:
                        print(f"Error fetching specific paper: {e}")
                        # Fall back to regular query if direct fetch fails
            
            # Regular query if not a URL or direct fetch failed
            papers = self.arxiv_full_text_fetcher.fetch_arxiv_full_text_from_query(state["search_term"], number_of_papers)
            self.arxiv_full_text_db.add_papers(papers)
            return {"arxiv_api_papers_info": papers}

        else:
            print("Not doing task 3!")
            return {"arxiv_api_papers_info": []}
        
    def summarize_abstract_node(self, state: ChatbotState):
        number_of_bullets = 5

        if state["new_tasks"][3]["task"] != "None":
            print("Doing task 4!")

            abstract_summaries = []
            for paper in state["query_vector_db_info"]:
                abstract_summary_prompt = f"""
                Here is the goal of the user: {state["user_request"].request}
                Here is the abstract: {paper.get('abstract', '')}
                Summarize the abstract in {number_of_bullets} detailed and technical bullet points that is less than 2 sentences each"""
        
                abstract_summary = call_llm(abstract_summary_prompt)
                abstract_summaries.append(abstract_summary.content)

            return {"summarize_abstract_info": abstract_summaries}

        else:
            print("Not doing task 4!")
            return {"summarize_abstract_info": []}
        
    def summarize_paper_node(self, state: ChatbotState):
        number_of_bullets = 10
        chunk_size = 10000

        if state["new_tasks"][4]["task"] != "None":
            print("Doing task 5!")

            paper_summaries = []
            for paper in state["query_vector_db_info"]:
                # Check if full_text exists in the paper
                if "full_text" not in paper and "content" not in paper:
                    print(f"Skipping paper without full text: {paper.get('title', 'Unknown title')}")
                    continue
                    
                # Get full text - might be in 'full_text' or 'content' key depending on source
                full_text = paper.get("full_text", paper.get("content", ""))
                if not full_text:
                    print(f"Empty full text for paper: {paper.get('title', 'Unknown title')}")
                    continue
                    
                # Split into chunks
                words = full_text.split()
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks.append(chunk)

                # Summarize each chunk
                chunk_summaries = []
                for chunk in chunks:
                    chunk_summary_prompt = f"""
                    Here is the goal of the user: {state["user_request"].request}
                    Here is the title of the paper: {paper.get("title", "Unknown title")}
                    Here is a section of the paper: {chunk}
                    Summarize this section in {number_of_bullets} detailed and technical bullet points that is less than 2 sentences each. Do not include any other text than the bullet points."""
            
                    chunk_summary = call_llm(chunk_summary_prompt)
                    chunk_summaries.append(chunk_summary.content)

                # Combine chunk summaries
                paper_summaries.append("\n\n".join(chunk_summaries))

            return {"summarize_paper_info": paper_summaries}

        else:
            print("Not doing task 5!")
            return {"summarize_paper_info": []}
        
    def compress_summaries_node(self, state: ChatbotState):
        if state["new_tasks"][5]["task"] != "None":
            print("Doing task 6!")

            compression_prompt = f"""
            Here is the goal of the user: {state["user_request"].request}
            Here is the summary of the abstract: {state["summarize_abstract_info"]}
            Here is the summary of the paper: {state["summarize_paper_info"]}
            Compress the summaries into a single summary in less than 1000 words. Use bullet points as much as possible. Be extremely technical, cogent, and convincing."""

            compressed_summary = call_llm(compression_prompt)
            print("Compressed summary: ", compressed_summary.content)
            return {"compress_summaries_info": compressed_summary.content}

        else:
            print("Not doing task 6!")
            return {"compress_summaries_info": []}
        
    def final_task_node(self, state: ChatbotState):
        """Generate a response based on the specific type of query"""
        if not state.get("is_new_thread", True):
            # For follow-up queries, analyze the type of question
            query_type_prompt = f"""
            Analyze the user's query to determine what type of information they're seeking:
            
            Query: {state["user_request"].request}
            
            Categorize this query into one of these types:
            1. SUMMARY - Wants a summary of content
            2. EXPLANATION - Wants something explained
            3. FACTUAL - Wants specific facts (authors, dates, citations)
            4. COMPARISON - Wants to compare aspects
            5. METHODOLOGY - Wants to understand methods/techniques
            
            Return ONLY the category name.
            """
            
            query_type = call_llm(query_type_prompt)
            query_category = query_type.content.strip().upper()
            
            # Formulate response based on query type
            response_prompt = f"""
            User's query: {state["user_request"].request}
            Query type: {query_category}
            Retrieved content: {state["query_vector_db_info"]}
            
            Generate a response that:
            1. Directly addresses the specific type of question asked
            2. Uses only information from the retrieved content
            3. Maintains technical accuracy and detail
            4. Cites specific papers when relevant
            
            Format the response appropriately for the query type:
            - SUMMARY: Use bullet points for key points
            - EXPLANATION: Step-by-step explanation
            - FACTUAL: Direct, concise statement of facts
            - COMPARISON: Clear parallel structure
            - METHODOLOGY: Technical detail with examples
            """
            
            response = call_llm(response_prompt)
            return {"final_task_info": response.content}
        else:
            # For initial queries, analyze papers and provide overview
            papers = state.get("query_vector_db_info", [])
            if not papers:
                return {"final_task_info": "I couldn't find any relevant papers for your query. Please try a different search term or check if the arXiv API is accessible."}
            
            initial_analysis_prompt = f"""
            User's request: {state["user_request"].request}
            Papers found: {papers}
            
            Provide a comprehensive initial analysis of these papers that includes:
            1. Overview of the main topics and themes
            2. Key findings or contributions
            3. Important methodologies or approaches
            4. Relevant authors and citations
            5. How these papers relate to the user's request
            
            Format the response in a clear, structured way using bullet points where appropriate.
            Be technical and precise, but ensure the information is accessible.
            """
            
            analysis = call_llm(initial_analysis_prompt)
            return {"final_task_info": analysis.content}

# Nodes
def review_tasks(state: ChatbotState):
    """
    Review the tasks based on whether this is a first message or follow-up.
    """
    is_new_thread = state.get("is_new_thread", True)
    
    # For follow-up messages, ALWAYS use follow-up workflow
    if not is_new_thread:
        print("Using follow-up workflow")
        return {
            "new_tasks": [
                {
                    "task": "Query understanding",
                    "justification": "Understanding follow-up query about existing papers."
                },
                {
                    "task": "Retrieve relevant content",
                    "justification": "Retrieving relevant content from stored papers."
                },
                {
                    "task": "Answer user query",
                    "justification": "Generating targeted response based on retrieved content."
                }
            ]
        }
    
    # For new threads, use the paper fetching workflow
    print("Using new thread workflow")
    return {
        "new_tasks": [
            {
                "task": "Fetch papers from Arxiv API",
                "justification": "Initial paper fetch for new query."
            },
            {
                "task": "Store papers in local vector database",
                "justification": "Storing papers for future reference."
            },
            {
                "task": "Initial paper analysis",
                "justification": "Analyzing retrieved papers."
            }
        ]
    }


def run_v1(prompt: str, is_new_thread: bool = True):
    structured_llm = llm.with_structured_output(UserRequest)
    
    formatted_prompt = f"""
    The following contains a chat history followed by the user's current request.
    Please extract:
    1. ONLY the user's current request (their most recent message)
    2. The previous chat history as a list of messages
    
    Format your response as a structured object with 'request' and 'chat_history' fields.
    
    {prompt}
    """
    
    message = call_structured_llm(structured_llm, formatted_prompt)
    print("prompt_v1: ", prompt)
    print("message_v1: ", message)
    print("Is new thread: ", is_new_thread)
    
    chatbot_tools = ChatBotTools()
    chatbot_builder = StateGraph(ChatbotState)
    
    # Common initial nodes for both workflows
    chatbot_builder.add_node("extract_search_term", chatbot_tools.extract_search_term_node)
    chatbot_builder.add_node("check_url_in_input", chatbot_tools.check_url_in_input_node)
    chatbot_builder.add_node("review_tasks", review_tasks)
    
    if is_new_thread:
        # First message workflow
        chatbot_builder.add_node("fetch_arxiv_papers", chatbot_tools.fetch_arxiv_papers_node)
        chatbot_builder.add_node("store_papers", chatbot_tools.store_papers_node)
        chatbot_builder.add_node("query_vector_db", chatbot_tools.query_vector_db_node)
        chatbot_builder.add_node("initial_analysis", chatbot_tools.final_task_node)
        
        # Build edges for first message workflow
        chatbot_builder.add_edge(START, "extract_search_term")
        chatbot_builder.add_edge("extract_search_term", "check_url_in_input")
        chatbot_builder.add_edge("check_url_in_input", "review_tasks")
        chatbot_builder.add_edge("review_tasks", "fetch_arxiv_papers")
        chatbot_builder.add_edge("fetch_arxiv_papers", "store_papers")
        chatbot_builder.add_edge("store_papers", "query_vector_db")
        chatbot_builder.add_edge("query_vector_db", "initial_analysis")
        chatbot_builder.add_edge("initial_analysis", END)
    else:
        # Follow-up message workflow
        chatbot_builder.add_node("query_understanding", chatbot_tools.extract_search_term_node)  # Reuse for query understanding
        chatbot_builder.add_node("retrieve_content", chatbot_tools.query_vector_db_node)
        chatbot_builder.add_node("answer_query", chatbot_tools.final_task_node)
        
        # Build edges for follow-up workflow
        chatbot_builder.add_edge(START, "query_understanding")
        chatbot_builder.add_edge("query_understanding", "review_tasks")
        chatbot_builder.add_edge("review_tasks", "retrieve_content")
        chatbot_builder.add_edge("retrieve_content", "answer_query")
        chatbot_builder.add_edge("answer_query", END)

    chatbot = chatbot_builder.compile()
    
    state = chatbot.invoke({
        "user_request": message,
        "is_new_thread": is_new_thread
    })
    
    # Debug prints
    if is_new_thread:
        print("NEW THREAD results:")
        print("- Fetched papers:", len(state.get("fetch_arxiv_papers_info", [])))
        print("- Stored papers:", state.get("store_papers_info", ""))
    print("- Retrieved papers from DB:", len(state.get("query_vector_db_info", [])))
    
    return state["final_task_info"]


# # state = run_v1("I am reading a paper called Restructuring Vector Quantization with the Rotation Trick. Can you help me understand it?")
# state = run_v1("I want to learn more about https://arxiv.org/abs/2205.14135. Tell me aboout it.")
# print(state)

# First query to populate the database
# state = run_v1("Tell me about recent advancements in transformer models.")
# print("First query response:", state)

# # Follow-up queries that should use the vector database
# state = run_v1("What are the key efficiency improvements in transformer architectures?")
# print("\nFollow-up query response:", state)

# # Another follow-up focusing on a specific aspect
# state = run_v1("How do these transformer models handle attention mechanisms?")
# print("\nSecond follow-up response:", state)