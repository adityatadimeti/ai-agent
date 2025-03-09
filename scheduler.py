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

llm = ChatMistralAI(model="mistral-small-latest")

tasks_workflow = """
The user is asking you to do research on a specific topic. We have outlined the exact workflow for you to complete below:

1. Name of Task: Query the Arxiv API for abstracts
(Below is information about the task above, do not include it in the task description)
- Inputs: Search term, number of papers to return
- Outputs: List of abstracts
- Prerequisites: None

2. Name of Task: Query the Arxiv API for papers
(Below is information about the task above, do not include it in the task description)
- Inputs: Search term, number of papers to return
- Outputs: List of papers
- Prerequisites: None

3. Name of Task: Check local vector database for relevance
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

6. Name of Task: Get daily update of papers
(Below is information about the task above, do not include it in the task description)
- Inputs: None
- Outputs: List of papers
- Prerequisites: 1 OR 2
"""

TASKS = [
    "Query the Arxiv API for abstracts",
    "Query the Arxiv API for papers",
    "Check local vector database for relevance",
    "Summarize the paper",
    "Summarize the abstract",
    "Get daily update of papers"
]

class UserRequest(BaseModel):
    request: str = Field(None, description="Request that is what the user is asking you to do")
    tasks: List[str] = TASKS

# Graph state
class ChatbotState(TypedDict):
    user_request: UserRequest
    new_tasks: List[Dict[str, str]]

# Nodes
def review_tasks(state: ChatbotState):
    """Review the tasks and justification one by oneand make sure they are relevant to the user's request.
       If not relevant, remove the task and justification and give a reason for the removal.
       If need more tasks, add more tasks and give a justification for the new tasks."""
    new_tasks = []
    new_justifications = []
    for task in zip(state["user_request"].tasks):
        print("Task: ", task)
        new_task_msg = llm.invoke(
            f"""User's request: {state["user_request"].request}
                Review the task ({task}) and make sure they are relevant to the user's request. 
                Here are the task capabilities: {tasks_workflow}
                If relevant, return the task, return only the task in JSON format: {{"task": "Task name", "justification": "Justification for the task"}}
                If not relevant, return only in the JSON format: {{"task": "None", "justification": "Reason for the removal"}}"""
        )
        print("New task: ", new_task_msg.content)
        time.sleep(1)
    return {"new_tasks": new_tasks}

structured_llm = llm.with_structured_output(UserRequest)
message = structured_llm.invoke("I am researching on the topic of FlashAttention and softmax of the KV cache. Can you help me answer some questions about it?")

chatbot_builder = StateGraph(ChatbotState)
chatbot_builder.add_node("review_tasks", review_tasks)
chatbot_builder.add_edge(START, "review_tasks")
chatbot_builder.add_edge("review_tasks", END)

chatbot = chatbot_builder.compile()

state = chatbot.invoke({"user_request": message})
print(state)