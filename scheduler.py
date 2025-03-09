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
- Outputs: Summary of the paper
- Prerequisites: 1 OR 3

5. Name of Task: Summarize the abstract
(Below is information about the task above, do not include it in the task description)
- Inputs: Abstract
- Outputs: Summary of the abstract
- Prerequisites: 2 OR 3

6. Name of Task: Compress the summaries into a single summary
(Below is information about the task above, do not include it in the task description)
- Inputs: Summary of the paper, summary of the abstract
- Outputs: Compressed summary
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
        print("User's request: ", state)
        search_term_prompt = f"""
            User's request: {state["user_request"].request}
            Extract the one search term phrase from the user's request to search Arxiv to get relevant papers.
            Only output the search term, nothing else.
        """
        print("Search term prompt: ", search_term_prompt)
        search_term = call_llm(search_term_prompt)
        print("Search term: ", search_term.content)
        return {"search_term": search_term.content}
    
        
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
        if state["new_tasks"][2]["task"] != "None"and state["check_local_vector_database_info"] == []:
            print("Doing task 3!")

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
            for abstract in state["arxiv_api_abstracts_info"]:
                abstract_summary_prompt = f"""
                Here is the goal of the user: {state["user_request"].request}
                Here is the abstract: {abstract}
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
                    Summarize this section in {number_of_bullets} detailed and technical bullet points that is less than 2 sentences each. Do not include any other text than the bullet points."""
            
                    chunk_summary = call_llm(chunk_summary_prompt)
                    # print("Chunk summary: ", chunk_summary.content)
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
        if state["new_tasks"][5]["task"] == "None":
            print("Doing task 7!")
            print("User's request: ", state["user_request"].request)
            response = call_llm(state["user_request"].request)
            print("Response: ", response.content)
            return {"final_task_info": response.content}

        else:
            print("Not doing task 7!")
            return {"final_task_info": state["compress_summaries_info"]}

# Nodes
def review_tasks(state: ChatbotState):
    """Review the tasks and justification one by one and make sure they are relevant to the user's request.
       If not relevant, remove the task and justification and give a reason for the removal.
       If need more tasks, add more tasks and give a justification for the new tasks."""
    new_tasks = []
    for task in TASKS:
        print("Task: ", task)
        new_task_msg =  call_llm(
            f"""User's request: {state["user_request"].request}
                Review the task ({task}) and make sure they are relevant to the user's request. 
                Here are the task capabilities: {tasks_workflow}
                If relevant, return the task, return only the task in JSON format: {{"task": "{task}", "justification": "Justification for the task"}}
                If not relevant, return only in the JSON format: {{"task": "None", "justification": "Reason for the removal"}}"""
        )
        print("New task: ", new_task_msg.content)
        # Strip any markdown formatting and extract just the JSON content
        

        double_check_prompt = f"""
        User's request: {state["user_request"].request}
        Task: {task}
        Here is the task capabilities: {tasks_workflow}
        Here is the new task: {new_task_msg.content}
        Are you sure about this skipping this task in terms of completing the user's request? 
        If you think we should do this task, return only the task in the JSON format: {{"task": "{task}", "justification": "Justification for the task"}}
        If you are sure we should not do this task, return only return the task in the JSON format: {{"task": "None", "justification": "Reason for the removal"}}
        """
        double_check = call_llm(double_check_prompt)
        print("Double check: ", double_check.content)
        content = double_check.content

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        try:
            task_json = json.loads(content)
            new_tasks.append(task_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            # Skip invalid JSON
            continue
        time.sleep(1)
    return {"new_tasks": new_tasks}


def run_v1(prompt: str):
    structured_llm = llm.with_structured_output(UserRequest)
    message = call_structured_llm(structured_llm, prompt)
    chatbot_tools = ChatBotTools()

    chatbot_builder = StateGraph(ChatbotState)
    chatbot_builder.add_node("extract_search_term", chatbot_tools.extract_search_term_node)
    chatbot_builder.add_node("review_tasks", review_tasks)
    chatbot_builder.add_node("check_local_vector_database", chatbot_tools.check_local_vector_database_node)
    chatbot_builder.add_node("query_arxiv_api_abstracts", chatbot_tools.query_arxiv_api_abstracts_node)
    chatbot_builder.add_node("query_arxiv_api_papers", chatbot_tools.query_arxiv_api_papers_node)
    chatbot_builder.add_node("summarize_abstract", chatbot_tools.summarize_abstract_node)
    chatbot_builder.add_node("summarize_paper", chatbot_tools.summarize_paper_node)
    chatbot_builder.add_node("compress_summaries", chatbot_tools.compress_summaries_node)
    chatbot_builder.add_node("final_task", chatbot_tools.final_task_node)

    chatbot_builder.add_edge(START, "extract_search_term")
    chatbot_builder.add_edge("extract_search_term", "review_tasks")
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


# run_v1("I am reading a paper called Restructuring Vector Quantization with the Rotation Trick. Can you help me understand it?")
# print(state)