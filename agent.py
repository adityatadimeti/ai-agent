import os
from mistralai import Mistral
import discord
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

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

    async def run(self, message: discord.Message):
        # Only process arxiv command
        if not message.content.startswith("!arxiv"):
            return None
            
        # Get papers and their abstracts
        papers_text = await self.fetch_arxiv_papers()
        
        # Create prompt for summarization
        prompt = f"Please provide a very concise summary (maximum 1500 characters) of the following recent arxiv papers. Focus on the most important papers and their key findings:\n\n{papers_text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content
