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
import os
import re
from langchain_core.tools import tool   
from vector_db.arxiv_vector_db import ArxivAbstractDB, ArxivFullTextDB, ArxivAbstractFetcher, ArxivFullTextFetcher


llm = ChatMistralAI(model="mistral-large-latest")

# Define tools
@tool
def extract_arxiv_query(arxiv_query: str) -> str:
    """Extract a detailed search query from the text of the user's message

    Args:
        arxiv_query: Suitable search term for Arxiv
    """
    return arxiv_query

@tool
def extract_arxiv_id(text: str) -> str:
    """Only extract arxiv ID from a URL if there is a URL in the text
    
    Args:
        text: Text containing an arxiv URL or ID
    """
    # Match common arxiv URL patterns and IDs
    pattern = r'(?:arxiv.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return "No arxiv ID found"



# Augment the LLM with tools
tools = [extract_arxiv_query, extract_arxiv_id]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant with helping deep research queries"
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """Performs the tool call"""
    print(state["messages"])
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def search_node(state: dict):
    """Performs the search on arxiv or our vector database"""
    print("Reached here!")
    # Initialize databases
    arxiv_abstract_db = ArxivAbstractDB()
    arxiv_full_text_db = ArxivFullTextDB()
    arxiv_abstract_fetcher = ArxivAbstractFetcher()
    
    # Get the search term from the last message
    messages = state["messages"]
    search_term = messages[-1].tool_calls[0]["args"]['arxiv_query']
    print(search_term)
    
    # First check abstract database
    _, is_relevant = arxiv_abstract_db.check_query_relevance(search_term)
    
    if not is_relevant:
        # If no hits in abstract DB, search arxiv directly
        print("Searching arxiv directly")
        print(search_term)
        papers = arxiv_abstract_fetcher.fetch_arxiv_abstracts(search_term)
        print(papers)
        arxiv_abstract_db.add_abstracts(papers)
        print("Added abstracts to abstract DB")
        return {"messages": papers}
    
    # If hits found in abstract DB, return those results
    print("Searching abstract DB")
    results = arxiv_abstract_db.query(search_term)
    return {"messages": results}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", "search", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    print("Last message: ", last_message.tool_calls)
    if last_message.tool_calls[0]["name"] == "extract_arxiv_query":
        print("Search")
        return "search"
    elif last_message.tool_calls:
        print("Action")
        return "Action"

    # Otherwise, we stop (reply to the user)
    print("End")
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)
agent_builder.add_node("search", search_node)
# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        "search": "search",
        END: END,   
    },
)
agent_builder.add_edge("environment", "llm_call")
agent_builder.add_edge("search", "llm_call")

# Compile the agent
agent = agent_builder.compile()



def main():
    # Show the agent
    # agent = ArxivFrontDesk
    # display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    # Invoke
    # messages = [HumanMessage(content="I am reading this paper, FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness https://arxiv.org/abs/2205.14135. Can you help me answer some questions about it?")]
    messages = [HumanMessage(content="I am researching on the topic of FlashAttention and softmax of the KV cache. Can you help me answer some questions about it?")]
    messages = agent.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    main()


    