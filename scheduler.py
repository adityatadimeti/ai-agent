import logging
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

# Initialize logger
logger = logging.getLogger("scheduler")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Silence httpx logs
logging.getLogger("httpx").disabled = True
logging.getLogger("discord.gateway").disabled = True
logging.getLogger("arxiv").disabled = True
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

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
The user is asking you to do research on a specific topic. We have outlined the exact workflow for you to complete below:

1. Name of Task: Check local vector database for relevance
(Below is information about the task above, do not include it in the task description)
- Inputs: Search term, number of papers to return
- Outputs: List of papers
- Prerequisites: None

2. Name of Task: Query the Arxiv API for abstracts
(Below is information about the task above, do not include it in the task description)
- Inputs: Search term, number of papers to return
- Outputs: List of abstracts
- Prerequisites: None

3. Name of Task: Query the Arxiv API for papers
(Below is information about the task above, do not include it in the task description)
- Inputs: Search term, number of papers to return
- Outputs: List of papers
- Prerequisites: None

4. Name of Task: Summarize the paper
(Below is information about the task above, do not include it in the task description)
- Inputs: Paper title, paper abstract
- Outputs: Summary of the paper, citations
- Prerequisites: 1 OR 3

5. Name of Task: Summarize the abstract
(Below is information about the task above, do not include it in the task description)
- Inputs: Abstract
- Outputs: Summary of the abstract, citations
- Prerequisites: 2 OR 3

6. Name of Task: Compress the summaries into a single summary
(Below is information about the task above, do not include it in the task description)
- Inputs: Summary of the paper, summary of the abstract
- Outputs: Compressed summary, citations
- Prerequisites: 4 OR 5
- SHOULD ALWAYS BE COMPLETED
"""

TASKS = [
    "Check local vector database for relevance",
    "Query the Arxiv API for abstracts",
    "Query the Arxiv API for papers",
    "Summarize the paper",
    "Summarize the abstract",
    "Compress the summaries into a single summary"
]

class UserRequest(BaseModel):
    request: str = Field(None, description="Request that is what the user is asking you to do")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="History of previous chat messages")

# Graph state
class ChatbotState(TypedDict):
    search_term: str
    user_request: UserRequest
    new_tasks: List[Dict[str, str]]
    check_local_vector_database_info: List[Dict[str, str]]
    arxiv_api_abstracts_info: List[Dict[str, str]]
    arxiv_api_papers_info: List[Dict[str, str]]
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
        logger.info(f"Extracting search term from request: {state['user_request'].request}")
        search_term_prompt = f"""
            User's request: {state["user_request"].request}
            Extract the one search term phrase from the user's request to search Arxiv to get relevant papers.
            Only output the search term, nothing else.
        """
        search_term = call_llm(search_term_prompt)
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

    def check_local_vector_database_node(self, state: ChatbotState):
        """Check the local vector database for relevance"""
        number_of_papers = 5
        
        if state["new_tasks"][0]["task"] != "None":
            logger.info(f"TASK 1: Checking local vector database for search term: {state['search_term']}")
            
            embellished_search_term = call_llm(
                f"""
                User's request: {state["user_request"].request}
                Search term: {state["search_term"]}
                Given this short search term, write a short abstract (< 200 words) about it so that we can embed the paragraph and search our vector database of abstract s-- In other words, you are embelishing the query for better results in vector search"
                """
            )

            _, is_relevant_abstract = self.arxiv_abstract_db.check_query_relevance(embellished_search_term.content, number_of_papers)
            _, is_relevant_full_text = self.arxiv_full_text_db.check_query_relevance(embellished_search_term.content, number_of_papers)
            
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
            logger.info("TASK 1: Skipping local vector database check - task marked as not needed")
            return {"check_local_vector_database_info": []}
        
    def query_arxiv_api_abstracts_node(self, state: ChatbotState):
        """Check the Arxiv API for relevance"""
        number_of_papers = 5
        
        if state["new_tasks"][1]["task"] != "None" and state["check_local_vector_database_info"] == []:
            logger.info(f"TASK 2: Fetching {number_of_papers} abstracts from Arxiv API for search term: {state['search_term']}")
            
            abstracts = self.arxiv_abstract_fetcher.fetch_arxiv_abstracts(state["search_term"], number_of_papers)
            self.arxiv_abstract_db.add_abstracts(abstracts)

            return {"arxiv_api_abstracts_info": abstracts}

        else:
            logger.info("TASK 2: Skipping Arxiv API abstract fetch - task marked as not needed")
            return {"arxiv_api_abstracts_info": []}
        
    def query_arxiv_api_papers_node(self, state: ChatbotState):
        """Check the Arxiv API for relevance"""
        number_of_papers = 5
        
        if state["new_tasks"][2]["task"] != "None"and state["check_local_vector_database_info"] == []:
            logger.info(f"TASK 3: Fetching {number_of_papers} papers from Arxiv API for search term: {state['search_term']}")

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
                        # self.arxiv_full_text_db.add_papers(papers)
                        return {"arxiv_api_papers_info": papers}
                    except Exception as e:
                        print(f"Error fetching specific paper: {e}")
                        # Fall back to regular query if direct fetch fails
            
            # Regular query if not a URL or direct fetch failed
            papers = self.arxiv_full_text_fetcher.fetch_arxiv_full_text_from_query(state["search_term"], number_of_papers)
            # TODO: WP 3/11/24: bypassing ChromaDB for now
            if False:
                self.arxiv_full_text_db.add_papers(papers)
            return {"arxiv_api_papers_info": papers}

        else:
            logger.info("TASK 3: Skipping Arxiv API paper fetch - task marked as not needed")
            return {"arxiv_api_papers_info": []}
        
    def summarize_abstract_node(self, state: ChatbotState):
        number_of_bullets = 5

        if state["new_tasks"][3]["task"] != "None":
            logger.info(f"TASK 4: Summarizing {number_of_bullets} abstracts")

            abstract_summaries = []
            for abstract in state["arxiv_api_abstracts_info"]:
                abstract_summary_prompt = f"""
                Here is the goal of the user: {state["user_request"].request}
                Here is the abstract: {abstract}
                Summarize the abstract in {number_of_bullets} detailed and technical bullet points that is less than 2 sentences each
                ALWAYS Cite your sources"""
        
                abstract_summary = call_llm(abstract_summary_prompt)
                abstract_summaries.append(abstract_summary.content)

            return {"summarize_abstract_info": abstract_summaries}

        else:
            logger.info("TASK 4: Skipping abstract summarization - task marked as not needed")
            return {"summarize_abstract_info": []}
        
    def summarize_paper_node(self, state: ChatbotState):
        number_of_bullets = 10
        chunk_size = 10000

        if state["new_tasks"][4]["task"] != "None":
            logger.info(f"TASK 5: Summarizing {number_of_bullets} papers")

            paper_summaries = []
            for paper in state["arxiv_api_papers_info"]:
                # Split abstract into 1000 word chunks
                full_text = paper["full_text"]
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
                    Here is the title of the paper: {paper["title"]}
                    Here is a section of the paper: {chunk}
                    Summarize this section in {number_of_bullets} detailed and technical bullet points that is less than 2 sentences each. Do not include any other text than the bullet points. ALWAYS Cite your sources"""
            
                    chunk_summary = call_llm(chunk_summary_prompt)
                    chunk_summaries.append(chunk_summary.content)

                # Combine chunk summaries
                paper_summaries.append("\n\n".join(chunk_summaries))

            return {"summarize_paper_info": paper_summaries}

        else:
            logger.info("TASK 5: Skipping paper summarization - task marked as not needed")
            return {"summarize_paper_info": []}
        
    def compress_summaries_node(self, state: ChatbotState):
        if state["new_tasks"][5]["task"] != "None":
            logger.info("TASK 6: Compressing summaries into a single summary")

            compression_prompt = f"""
            Here is the goal of the user: {state["user_request"].request}
            Here is the summary of the abstract: {state["summarize_abstract_info"]}
            Here is the summary of the paper: {state["summarize_paper_info"]}
            Compress the summaries into a single summary in less than 1000 words. Use bullet points as much as possible. Be extremely technical, cogent, and convincing. ALWAYS Cite your sources"""

            compressed_summary = call_llm(compression_prompt)
            print("Compressed summary: ", compressed_summary.content)
            return {"compress_summaries_info": compressed_summary.content}

        else:
            logger.info("TASK 6: Skipping summary compression - task marked as not needed")
            return {"compress_summaries_info": []}
        
    def final_task_node(self, state: ChatbotState):
        if state["new_tasks"][5]["task"] == "None":
            logger.info(f"TASK 7: Final task - formatting response")
            
            # Format chat history if it exists
            chat_history_text = ""
            has_chat_history = hasattr(state["user_request"], "chat_history") and state["user_request"].chat_history
            if has_chat_history:
                chat_history_text = state["user_request"].chat_history
            
            # First determine if user is referencing previous chat
            reference_detection_prompt = f"""
            User's request: {state["user_request"].request}
            
            {chat_history_text if has_chat_history else "No chat history available."}
            
            Analyze the user's request and determine if they are referencing or following up on something from a previous conversation.
            Look for:
            - Pronouns without clear referents (it, that, those, etc.)
            - Questions that seem to continue a previous topic
            - Requests for clarification or elaboration without specifying what
            - Mentions of "earlier", "before", "you mentioned", etc.
            - Phrases such as "last chat", "previous chat", "previous conversation", etc.
            
            Return only "YES" if the user is likely referencing previous chat, or "NO" if it's a standalone request.
            """
            
            reference_detection = call_llm(reference_detection_prompt)
            is_referencing_history = "YES" in reference_detection.content.upper()
            
            # Generate appropriate response based on detection
            if is_referencing_history and has_chat_history:
                response_prompt = f"""                
                User's request: {state["user_request"].request}
                
                The user appears to be referencing your previous conversation. You have access to the following chat history to provide a contextually appropriate response.

                User's chat history (from earliest to latest): {chat_history_text}

                You have access to the above chat history, do not say that you do not have access to it.
                """
            else:
                response_prompt = f"""
                Remind the user that you are a bot tasked with helping research / technical questions and querying the Arxiv repository to answer them.
                If you end up at this step, it means that the user's request is not related to research or the Arxiv repository.
                Please mention that to them.
                
                User's request: {state["user_request"].request}
                """
            
            response = call_llm(response_prompt)
            return {"final_task_info": response.content}

        else:
            logger.info("TASK 7: Skipping final task - task marked as not needed")
            return {"final_task_info": state["compress_summaries_info"]}

# Nodes
def review_tasks(state: ChatbotState):
    """
       Review the tasks and justification one by one and make sure they are relevant to the user's request.
       If not relevant, remove the task and justification and give a reason for the removal.
       If need more tasks, add more tasks and give a justification for the new tasks."""
    new_tasks = []
    for task in TASKS:
        new_task_msg =  call_llm(
            f"""User's request: {state["user_request"].request}\
                Reminder, you are a bot tasked with helping research / technical questions and querying the Arxiv repository to answer them.
                Review the task ({task}) and make sure they are relevant to the user's request. 
                Here are the task capabilities: {tasks_workflow}
                If relevant, return the task, return only the task in JSON format: {{"task": "{task}", "justification": "Justification for the task"}}
                If not relevant, return only in the JSON format: {{"task": "None", "justification": "Reason for the removal"}}"""
        )
        logger.info(f"{task}")
        logger.info(f"Doing step: - {'Yes' if 'None' not in new_task_msg.content else 'No'}")
        # Strip any markdown formatting and extract just the JSON content
        

        double_check_prompt = f"""
        User's request: {state["user_request"].request}
        Task: {task}
        Here is the task capabilities: {tasks_workflow}
        Here is the new task: {new_task_msg.content}
        Reminder, you are a bot tasked with helping research / technical questions and querying the Arxiv repository to answer them.
        Are you sure about this skipping this task in terms of completing the user's request? 
        Example: Queries that ask you something that is not related to research or the Arxiv repository are not relevant, like "hi" or "please set up my user profile"
        If you think we should do this task, return only the task in the JSON format: {{"task": "{task}", "justification": "Justification for the task"}}
        If you are sure we should not do this task, return only return the task in the JSON format: {{"task": "None", "justification": "Reason for the removal"}}
        """
        double_check = call_llm(double_check_prompt)
        logger.info(f"Double check: {'Yes' if 'None' not in double_check.content else 'No'}")
        content = double_check.content

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        try:
            task_json = json.loads(content)
            new_tasks.append(task_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Warning: parsing JSON {e}")
            # Skip invalid JSON
            continue
        time.sleep(1)
    return {"new_tasks": new_tasks}


def run_v1(prompt: str):
    structured_llm = llm.with_structured_output(UserRequest)
    
    # Add instructions to extract just the current request
    formatted_prompt = f"""
    The following contains a chat history followed by the user's current request.
    Please extract:
    1. ONLY the user's current request (their most recent message)
    2. The previous chat history as a list of messages
    
    Format your response as a structured object with 'request' and 'chat_history' fields.
    
    {prompt}
    """
    
    message = call_structured_llm(structured_llm, formatted_prompt)
    chatbot_tools = ChatBotTools()

    chatbot_builder = StateGraph(ChatbotState)
    chatbot_builder.add_node("extract_search_term", chatbot_tools.extract_search_term_node)
    chatbot_builder.add_node("check_url_in_input", chatbot_tools.check_url_in_input_node)
    chatbot_builder.add_node("review_tasks", review_tasks)
    chatbot_builder.add_node("check_local_vector_database", chatbot_tools.check_local_vector_database_node)
    chatbot_builder.add_node("query_arxiv_api_abstracts", chatbot_tools.query_arxiv_api_abstracts_node)
    chatbot_builder.add_node("query_arxiv_api_papers", chatbot_tools.query_arxiv_api_papers_node)
    chatbot_builder.add_node("summarize_abstract", chatbot_tools.summarize_abstract_node)
    chatbot_builder.add_node("summarize_paper", chatbot_tools.summarize_paper_node)
    chatbot_builder.add_node("compress_summaries", chatbot_tools.compress_summaries_node)
    chatbot_builder.add_node("final_task", chatbot_tools.final_task_node)

    chatbot_builder.add_edge(START, "extract_search_term")
    chatbot_builder.add_edge("extract_search_term", "check_url_in_input")
    chatbot_builder.add_edge("check_url_in_input", "review_tasks")
    chatbot_builder.add_edge("review_tasks", "check_local_vector_database")
    chatbot_builder.add_edge("check_local_vector_database", "query_arxiv_api_abstracts")
    chatbot_builder.add_edge("query_arxiv_api_abstracts", "query_arxiv_api_papers")
    chatbot_builder.add_edge("query_arxiv_api_papers", "summarize_abstract")
    chatbot_builder.add_edge("summarize_abstract", "summarize_paper")
    chatbot_builder.add_edge("summarize_paper", "compress_summaries")
    chatbot_builder.add_edge("compress_summaries", "final_task")
    chatbot_builder.add_edge("final_task", END)

    chatbot = chatbot_builder.compile()

    state = chatbot.invoke({"user_request": message})
    return state["final_task_info"]


# state = run_v1("I am reading a paper called Restructuring Vector Quantization with the Rotation Trick. Can you help me understand it?")
# state = run_v1("I want to learn more about https://arxiv.org/abs/2205.14135. Tell me aboout it.")
# print(state)