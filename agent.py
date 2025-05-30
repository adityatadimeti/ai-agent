import os
from mistralai import Mistral
import discord
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from scheduler import run_v1, call_llm
import logging
from bot_utils import find_existing_thread, queue_message, queue_chunked_message

logger = logging.getLogger("agent")

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful assistant. Summarize the following arxiv papers in a clear and concise way, focusing on the key findings and implications."


class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        self.client = Mistral(api_key=MISTRAL_API_KEY)
    
    async def fetch_arxiv_papers(self):
        # Compute the date range for the past week
        end_date = datetime.utcnow().strftime("%Y%m%d")
        start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y%m%d")

        # Define the arXiv API URL - limit to 10 papers
        url = f"http://export.arxiv.org/api/query?search_query=lastUpdatedDate:[{start_date} TO {end_date}]&max_results=10"

        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        # Parse the XML response
        root = ET.fromstring(response.text)
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}

        # Combine all abstracts into one text
        combined_text = ""
        for entry in root.findall('arxiv:entry', namespace):
            title = entry.find('arxiv:title', namespace).text.strip()
            abstract = entry.find('arxiv:summary', namespace).text.strip()
            combined_text += f"Title: {title}\nAbstract: {abstract}\n\n"

        return combined_text

    async def fetch_arxiv_papers_by_topic(self, search_term):
        # Compute the date range for the past week
        end_date = datetime.utcnow().strftime("%Y%m%d")
        start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y%m%d")

        # URL encode the search term
        encoded_search = requests.utils.quote(search_term)
        
        # Define the arXiv API URL with search term and date range - limit to 10 papers
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_search}+AND+lastUpdatedDate:[{start_date} TO {end_date}]&max_results=10"

        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        # Parse the XML response
        root = ET.fromstring(response.text)
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}

        # Combine all abstracts into one text
        combined_text = ""
        for entry in root.findall('arxiv:entry', namespace):
            title = entry.find('arxiv:title', namespace).text.strip()
            abstract = entry.find('arxiv:summary', namespace).text.strip()
            combined_text += f"Title: {title}\nAbstract: {abstract}\n\n"

        return combined_text

    def run(self, message: discord.Message):
        response = run_v1(message.content)
        logger.info(f"Response: {response}")
        # if response:
        #     # Create a thread if needed, using create_task to not block
        #     # thread_name = "Profile Setup - " + message.author.name if "profile" in response.lower() else self.make_thread_name(message)
            
        #     # Check if the thread already exists - do this asynchronously
        #     # existing_thread = find_existing_thread(message.channel, thread_name)
            
        #     # # If the thread already exists, use it
        #     # if existing_thread:
        #     #     thread = existing_thread
        #     #     queue_message(message.channel, f"Let's discuss more in the previous thread {thread.mention}")
            
        #     # Send response in chunks using the rate-limited queue
        #     queue_chunked_message(outbound_message_queue, message, response)
            
        return response

    async def make_thread_name(self, message: discord.Message):
        """Extract the title of the thread from the user's request"""
        title_of_thread_prompt = f"""
            User's request: {message.content}
            Extract the title of the thread from the user's request that is succinct, less than 7 words.
            Only output the title of the thread, nothing else.
        """
        title_of_thread = call_llm(title_of_thread_prompt)
        logger.info(f"Title of thread: {title_of_thread.content}")
        return title_of_thread.content
    