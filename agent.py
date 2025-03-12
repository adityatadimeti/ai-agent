import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import discord
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import asyncio
import logging
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('agent')

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful research assistant."

# Track active searches by user ID
active_searches = {}

# Store completed search results by thread ID
completed_searches = {}

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = MistralClient(api_key=MISTRAL_API_KEY)
        logger.info("MistralAgent initialized")
    
    def validate_date_format(self, date_str):
        """Validate if a string is in YYYY-MM-DD format and is a valid date"""
        if not date_str:
            return False
            
        # Check format with regex
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return False
            
        # Check if it's a valid date
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
            
    def get_default_date_range(self):
        """Return default date range (all time)"""
        end_date = datetime.utcnow().strftime("%Y%m%d")
        # Default to 10 years ago
        start_date = (datetime.utcnow() - timedelta(days=3650)).strftime("%Y%m%d")
        return {
            "has_date_range": True,
            "start_date": start_date,
            "end_date": end_date
        }
    
    async def check_query_relevance(self, query):
        """Check if the query is relevant to academic research papers"""
        logger.info(f"Checking relevance of query: {query}")
        
        messages = [
            ChatMessage(role="system", content="You are an academic research assistant that helps find research papers. Your task is to determine if a query is relevant to academic research papers and scientific topics that could be found on ArXiv. You should be VERY strict and only approve queries that are clearly academic in nature. Personal questions, general knowledge questions, or questions about non-academic topics should be marked as irrelevant. Weather questions are only relevant if they specifically mention climate research or meteorological models. Respond with ONLY 'RELEVANT' or 'IRRELEVANT'."),
            ChatMessage(role="user", content=f"Query: {query}\n\nIs this query relevant to academic research papers that could be found on ArXiv? Answer ONLY with 'RELEVANT' or 'IRRELEVANT'.")
        ]
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            result = response.choices[0].message.content.strip().upper()
            logger.info(f"Query relevance check result: {result}")
            
            # Make sure we get a clear RELEVANT or IRRELEVANT (handle cases where model provides explanation)
            if result == "RELEVANT":
                return True
            elif result == "IRRELEVANT":
                # If deemed irrelevant, try interpreting the query first before rejecting
                logger.info(f"Query initially deemed irrelevant, attempting to interpret: {query}")
                return await self.try_interpret_query(query)
            else:
                # If the model didn't follow format exactly, try to interpret
                logger.warning(f"Unclear relevance result: {result}. Attempting to interpret.")
                return await self.try_interpret_query(query)
        except Exception as e:
            logger.error(f"Error checking query relevance: {str(e)}", exc_info=True)
            # For errors, try interpreting before rejecting
            return await self.try_interpret_query(query)
    
    async def try_interpret_query(self, query):
        """Try to interpret an ambiguous query in an academic context"""
        logger.info(f"Attempting to interpret query: {query}")
        
        messages = [
            ChatMessage(role="system", content="You are an expert academic research assistant with deep knowledge of computer science, AI, machine learning, and many other academic fields. Your task is to take a user query that might be ambiguous or informal and determine if it could be interpreted as related to academic research. Technical terms like 'KV Cache' (key-value cache in transformer models) or domain-specific jargon should be recognized as potentially academic. If there's ANY reasonable interpretation that could make this query relevant to academic papers, respond with 'ACADEMIC: interpretation'. If there's absolutely no way to interpret this as academic, respond with 'NON-ACADEMIC'."),
            ChatMessage(role="user", content=f"Query: '{query}'\n\nCould this be interpreted as an academic research topic? If yes, provide a brief academic interpretation prefixed with 'ACADEMIC:'. If no, just say 'NON-ACADEMIC'.")
        ]
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Query interpretation result: {result}")
            
            if result.upper().startswith("ACADEMIC:"):
                interpretation = result[9:].strip()  # Extract the interpretation part
                logger.info(f"Query interpreted academically as: {interpretation}")
                return True
            else:
                logger.info("Query could not be interpreted academically")
                return False
        except Exception as e:
            logger.error(f"Error interpreting query: {str(e)}", exc_info=True)
            # In case of error, default to treating as relevant to avoid false negatives
            return True
    
    async def distill_search_query_excluding_dates(self, user_message):
        """Extract a clean, academic search query from conversational text, excluding date references"""
        logger.info(f"Distilling search query (excluding dates) from: {user_message}")
        
        messages = [
            ChatMessage(role="system", content="""
You are an expert research librarian. Your task is to extract a clear, precise academic search query from a user's message.

IMPORTANT INSTRUCTIONS:
1. Remove any conversational elements, greetings, or irrelevant text
2. Focus on identifying the core academic/scientific topic
3. EXCLUDE all date references and time periods (e.g., 'last 2 years', 'since 2019', '2020 to 2022')
4. Return ONLY the distilled search terms without any prefixes, explanation, or punctuation
5. If multiple academic topics are mentioned, focus on the most prominent one

Examples:
Input: "Can you find me papers about quantum computing from 2020 to 2022?"
Output: "quantum computing"

Input: "I'd like to see research on climate change published in the last 5 years"
Output: "climate change"

Input: "Looking for recent papers on KV Cache implementations since 2021"
Output: "KV Cache implementations"
"""),
            ChatMessage(role="user", content=f"User message: '{user_message}'\n\nPlease extract a clear academic search query, excluding any date references, from this message. Return ONLY the search terms.")
        ]
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            distilled_query = response.choices[0].message.content.strip()
            logger.info(f"Distilled search query (excluding dates): '{distilled_query}' from original: '{user_message}'")
            return distilled_query
        except Exception as e:
            logger.error(f"Error distilling search query: {str(e)}", exc_info=True)
            # Fallback to basic distillation if there's an error
            return await self.distill_search_query(user_message)
    
    async def distill_search_query(self, user_message):
        """Extract a clean, academic search query from conversational text"""
        logger.info(f"Distilling search query from: {user_message}")
        
        messages = [
            ChatMessage(role="system", content="You are an expert research librarian. Your task is to extract a clear, precise academic search query from a user's message. Remove any conversational elements, greetings, or irrelevant text. Focus on identifying the core academic/scientific topic. Return ONLY the distilled search terms without any prefixes, explanation, or punctuation. If multiple topics are mentioned, focus on the most prominent one."),
            ChatMessage(role="user", content=f"User message: '{user_message}'\n\nPlease extract a clear academic search query from this message. Return ONLY the search terms.")
        ]
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            distilled_query = response.choices[0].message.content.strip()
            logger.info(f"Distilled search query: '{distilled_query}' from original: '{user_message}'")
            return distilled_query
        except Exception as e:
            logger.error(f"Error distilling search query: {str(e)}", exc_info=True)
            # Return the original message if we encounter an error
            return user_message
    
    
    
    async def get_academic_suggestion(self, query):
        """Generate an academic suggestion for a non-academic query"""
        logger.info(f"Generating academic suggestion for: {query}")
        
        messages = [
            ChatMessage(role="system", content="You are an expert academic research assistant. Your task is to take a user query that might not be academic in nature, and suggest a related academic research topic that would be appropriate for searching on ArXiv. Focus on converting casual language, questions about technology, or general inquiries into specific academic research topics. For example, 'good AI models' could become 'comparative analysis of large language models'. If there's absolutely no way to relate the query to academic research, respond with 'NO_SUGGESTION'."),
            ChatMessage(role="user", content=f"Query: '{query}'\n\nWhat academic research topic might this person be trying to ask about? Provide a specific, well-formulated academic research topic. If it's impossible to relate to academic research, respond with ONLY 'NO_SUGGESTION'.")
        ]
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Academic suggestion result: {result}")
            
            if result == "NO_SUGGESTION":
                return None
            else:
                # Return the academic suggestion
                return result
        except Exception as e:
            logger.error(f"Error generating academic suggestion: {str(e)}", exc_info=True)
            return None
    
    async def clarify_query(self, user_query):
        """Clarify the user's search query"""
        logger.info(f"Clarifying query: {user_query}")
        
        messages = [
            ChatMessage(role="system", content="You are a helpful research assistant. Your task is to clarify what the user is trying to find in recent research papers. Ask one specific question to help narrow down their search intent."),
            ChatMessage(role="user", content=f"I want to search for recent research papers about: {user_query}")
        ]
        
        logger.debug(f"Sending messages to Mistral: {[{m.role: m.content} for m in messages]}")
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            logger.debug(f"Full Mistral response: {json.dumps(response.model_dump(), default=str)}")
            clarification = response.choices[0].message.content
            logger.info(f"Received clarification: {clarification[:100]}...")
            
            # Add explicit date range instructions
            date_instructions = """

Please also specify a date range for your search in YYYY-MM-DD format:
- Start date (e.g., 2020-01-01)
- End date (e.g., 2023-12-31)

Or type "all time" if you want to search across all available papers."""
            
            clarification += date_instructions
            logger.info("Added date range instructions to clarification")
            
            return clarification
        except Exception as e:
            logger.error(f"Error getting clarification: {str(e)}", exc_info=True)
            raise
    
    async def parse_date_input(self, message_content):
        """Parse a user's date input message to extract dates in YYYY-MM-DD format"""
        logger.info(f"Parsing date input: {message_content}")
        
        # Check for "all time" option
        if "all time" in message_content.lower():
            logger.info("User requested all time search")
            return self.get_default_date_range()
            
        # Extract dates with regex
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, message_content)
        
        if len(dates) >= 2:
            # We have at least two dates, assume first is start and second is end
            start_date = dates[0]
            end_date = dates[1]
            
            # Validate dates
            if not self.validate_date_format(start_date) or not self.validate_date_format(end_date):
                logger.warning(f"Invalid date format found: {start_date} or {end_date}")
                return None
                
            # Convert to YYYYMMDD format for ArXiv
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Make sure start date is before end date
            if start_date_obj > end_date_obj:
                start_date_obj, end_date_obj = end_date_obj, start_date_obj
                
            # Make sure end date is not in the future
            today = datetime.utcnow()
            if end_date_obj > today:
                end_date_obj = today
                
            return {
                "has_date_range": True,
                "start_date": start_date_obj.strftime("%Y%m%d"),
                "end_date": end_date_obj.strftime("%Y%m%d")
            }
        elif len(dates) == 1:
            # Just one date found, check if it looks like a start date or end date
            date_str = dates[0]
            
            if not self.validate_date_format(date_str):
                logger.warning(f"Invalid date format found: {date_str}")
                return None
                
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            today = datetime.utcnow()
            
            # If the date is within the last year, assume it's a start date
            if date_obj > (today - timedelta(days=365)):
                # Use as start date, end date is today
                return {
                    "has_date_range": True,
                    "start_date": date_obj.strftime("%Y%m%d"),
                    "end_date": today.strftime("%Y%m%d")
                }
            else:
                # Use as start date, end date is today
                return {
                    "has_date_range": True,
                    "start_date": date_obj.strftime("%Y%m%d"),
                    "end_date": today.strftime("%Y%m%d")
                }
        
        # No valid dates found
        return None
    
    async def fetch_arxiv_papers_by_topic(self, search_term, custom_date_info=None):
        """Search ArXiv papers by the given search term with optional date range"""
        logger.info(f"Searching ArXiv for: {search_term}")
        
        # Process date range
        if custom_date_info and custom_date_info.get("has_date_range", False):
            # Extract dates from the date info
            start_date = custom_date_info.get("start_date")
            end_date = custom_date_info.get("end_date")
            
            # Validate dates
            is_valid = True
            current_date = datetime.utcnow().strftime("%Y%m%d")
            
            # Set defaults if needed
            if not start_date:
                # If no start date but we have an end date, use a year before end date
                if end_date:
                    try:
                        end_dt = datetime.strptime(end_date, "%Y%m%d")
                        start_date = (end_dt - timedelta(days=365)).strftime("%Y%m%d")
                    except ValueError:
                        is_valid = False
                else:
                    # If no dates at all, default to 10 years ago
                    start_date = (datetime.utcnow() - timedelta(days=3650)).strftime("%Y%m%d")
            
            if not end_date:
                # Default end date to current date
                end_date = current_date
            
            # Check date validity
            try:
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                
                # Ensure start date is before end date
                if start_dt > end_dt:
                    logger.warning(f"Invalid date range: start date {start_date} is after end date {end_date}")
                    is_valid = False
                
                # Ensure end date is not in the future
                current_dt = datetime.utcnow()
                if end_dt > current_dt:
                    logger.warning(f"End date {end_date} is in the future, adjusting to current date")
                    end_date = current_date
                
            except ValueError:
                logger.warning(f"Invalid date format: {start_date} or {end_date}")
                is_valid = False
            
            if not is_valid:
                logger.warning("Invalid date range specified. Using full history.")
                # Fall back to wide date range (10 years)
                start_date = (datetime.utcnow() - timedelta(days=3650)).strftime("%Y%m%d")
                end_date = current_date
                logger.info(f"Using fallback date range: {start_date} to {end_date}")
        else:
            # Default to last week if no custom date range
            end_date = datetime.utcnow().strftime("%Y%m%d")
            start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y%m%d")
            logger.info(f"Using default date range (last week): {start_date} to {end_date}")

        logger.info(f"Date range for search: {start_date} to {end_date}")
        
        # URL encode the search term
        encoded_search = requests.utils.quote(search_term)
        
        # Define ArXiv API URL with search term and date range
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_search}+AND+lastUpdatedDate:[{start_date} TO {end_date}]&max_results=10"
        logger.debug(f"ArXiv API URL: {url}")

        # Simulate longer search to demonstrate rate limiting
        await asyncio.sleep(2)
        
        # Make the API request
        try:
            logger.info(f"Making request to ArXiv API")
            response = requests.get(url)
            response.raise_for_status()
            logger.debug(f"ArXiv response status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching from ArXiv: {str(e)}", exc_info=True)
            raise

        # Parse the XML response
        root = ET.fromstring(response.text)
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}

        # Combine all abstracts into one text
        combined_text = ""
        paper_count = 0
        for entry in root.findall('arxiv:entry', namespace):
            paper_count += 1
            title = entry.find('arxiv:title', namespace).text.strip()
            abstract = entry.find('arxiv:summary', namespace).text.strip()
            
            # Try to find authors
            authors = []
            for author in entry.findall('arxiv:author', namespace):
                name = author.find('arxiv:name', namespace)
                if name is not None and name.text:
                    authors.append(name.text)
            author_str = ", ".join(authors) if authors else "Unknown"
            
            # Try to find link
            link = None
            for link_elem in entry.findall('arxiv:link', namespace):
                if link_elem.get('title') == "pdf":
                    link = link_elem.get('href')
                    break
            link_str = link if link else entry.find('arxiv:id', namespace).text
            
            logger.debug(f"Found paper: {title}")
            combined_text += f"Title: {title}\nAuthors: {author_str}\nLink: {link_str}\nAbstract: {abstract}\n\n"

        logger.info(f"Found {paper_count} papers matching the query in date range {start_date} to {end_date}")
        
        if paper_count == 0:
            date_info = f" between {start_date[:4]}-{start_date[4:6]}-{start_date[6:]} and {end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            return f"No papers found matching your criteria{date_info}."
            
        return combined_text
    
    async def summarize_papers(self, papers_text, original_query):
        """Summarize the papers based on the user's query"""
        logger.info(f"Summarizing papers for query: {original_query}")
        
        messages = [
            ChatMessage(role="system", content="You are a helpful research assistant. Summarize the following research papers in relation to the user's query. Focus on the most relevant findings. Format your response nicely with markdown."),
            ChatMessage(role="user", content=f"Query: {original_query}\n\nPapers:\n{papers_text}")
        ]
        
        logger.debug(f"Sending summarization request to Mistral")
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            summary = response.choices[0].message.content
            logger.debug(f"Received summary of length: {len(summary)}")
            logger.info(f"Summary preview: {summary[:100]}...")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing papers: {str(e)}", exc_info=True)
            raise
    
    async def answer_question_about_papers(self, question, papers_text, original_query):
        """Answer a question about previously found papers"""
        logger.info(f"Answering question about papers: {question}")
        
        messages = [
            ChatMessage(role="system", content="You are a helpful research assistant. Answer the user's question based on the provided research papers. Be thorough and accurate."),
            ChatMessage(role="user", content=f"These papers were found for the query: {original_query}\n\nPapers:\n{papers_text}\n\nQuestion: {question}")
        ]
        
        logger.debug(f"Sending question to Mistral about papers")
        
        try:
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            answer = response.choices[0].message.content
            logger.debug(f"Received answer of length: {len(answer)}")
            logger.info(f"Answer preview: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            raise
    
    async def run(self, message, is_follow_up=False, original_query=None):
        """Main processing function with rate limiting"""
        user_id = message.author.id
        channel_id = message.channel.id
        
        logger.info(f"Processing message from user {user_id}: '{message.content[:50]}...'")
        logger.info(f"is_follow_up: {is_follow_up}, original_query: {original_query}")
        
        # Check if this is a follow-up question about completed search results
        if isinstance(message.channel, discord.Thread) and channel_id in completed_searches:
            logger.info(f"Found existing research results for thread {channel_id}")
            # Check if the follow-up question is relevant to research
            is_relevant = await self.check_query_relevance(message.content)
            if not is_relevant:
                logger.info(f"Irrelevant follow-up question detected: {message.content}")
                return "I'm an ArXiv research assistant designed to help with academic research papers. Your question doesn't seem related to research papers. Could you please ask a question related to the research papers we found?"
            
            # This is a question about previous search results
            papers_text = completed_searches[channel_id]['papers']
            search_query = completed_searches[channel_id]['query']
            logger.info(f"Answering question about previous search: {search_query}")
            answer = await self.answer_question_about_papers(message.content, papers_text, search_query)
            return answer
        
        # If this is not a follow-up and the user has an active search, return a message
        if not is_follow_up and user_id in active_searches:
            logger.info(f"User {user_id} has an active search already")
            return "I'm still searching for papers from your previous query. Please wait a moment."
        
        # Handle new search request
        if not is_follow_up:
            logger.info(f"Processing new search request from user {user_id}")
            
            # Check if the query is relevant to research
            is_relevant = await self.check_query_relevance(message.content)
            if not is_relevant:
                logger.info(f"Irrelevant query detected: {message.content}")
                
                # Try to get a suggested academic reformulation
                academic_suggestion = await self.get_academic_suggestion(message.content)
                
                if academic_suggestion:
                    return f"I'm an ArXiv research assistant designed to help with academic research papers. Your query doesn't seem directly related to academic research. Did you mean to ask about '{academic_suggestion}'? If so, please rephrase your question to be more specific to academic research."
                else:
                    return "I'm an ArXiv research assistant designed to help you find academic research papers. Your query doesn't seem related to academic research. Could you please ask about a research topic? For example, you could ask about 'quantum computing', 'climate modeling', or 'neural networks'."
            
            # Store the original query (no distillation at this stage)
            active_searches[user_id] = {
                'status': 'awaiting_clarification', 
                'original_query_raw': message.content,
                'date_format_attempts': 0  # Add counter for date format attempts
            }
            logger.info(f"Set status for user {user_id} to 'awaiting_clarification'")
            
            # Return clarification question
            clarification = await self.clarify_query(message.content)
            return clarification
        
        # Handle follow-up (clarification response)
        elif is_follow_up and original_query:
            # Update status to searching
            logger.info(f"Processing clarification response for query: {original_query}")
            
            # Check if the clarification is relevant
            is_relevant = await self.check_query_relevance(message.content)
            if not is_relevant:
                del active_searches[user_id]
                logger.info(f"Irrelevant clarification detected: {message.content}")
                
                # Try to get a suggested academic reformulation
                academic_suggestion = await self.get_academic_suggestion(message.content)
                
                if academic_suggestion:
                    return f"I'm an ArXiv research assistant designed to help with academic research papers. Your clarification doesn't seem directly related to academic research. Did you mean to ask about '{academic_suggestion}'? If so, please rephrase your clarification to be more specific to academic research."
                else:
                    return "I'm an ArXiv research assistant designed to help with academic research papers. Your clarification doesn't seem related to academic research. Please provide a clarification that relates to research papers."
            
            # Get original raw query
            original_query_raw = active_searches.get(user_id, {}).get('original_query_raw', original_query)
            clarification_raw = message.content
            
            # Check for dates in the clarification message
            date_info = await self.parse_date_input(clarification_raw)
            
            # If date format is invalid and we haven't exceeded attempts, ask for correct format
            current_attempts = active_searches.get(user_id, {}).get('date_format_attempts', 0)
            
            if date_info is None and current_attempts < 2:
                # Increment attempt counter
                active_searches[user_id]['date_format_attempts'] = current_attempts + 1
                logger.info(f"Invalid date format detected, attempts: {current_attempts + 1}")
                
                return """Please provide dates in the correct format:
                
Start date: YYYY-MM-DD (e.g., 2020-01-01)
End date: YYYY-MM-DD (e.g., 2023-12-31)

Or simply type "all time" if you want to search across all available papers."""
            
            # If still no valid date info, use default (all time)
            if date_info is None:
                logger.info("Using default date range after failed attempts")
                date_info = self.get_default_date_range()
                
            # Distill the query without dates
            combined_raw_query = f"{original_query_raw} {clarification_raw}"
            distilled_query = await self.distill_search_query_excluding_dates(combined_raw_query)
            logger.info(f"Distilled combined query: '{distilled_query}'")
            
            # Update active searches status
            active_searches[user_id] = {
                'status': 'searching',
                'original_query_raw': original_query_raw,
                'clarification_raw': clarification_raw,
                'combined_raw_query': combined_raw_query,
                'distilled_query': distilled_query
            }
            
            # Fetch papers based on clarified query
            await message.channel.send("Searching for relevant papers... This may take a moment.")
            
            # Search for papers
            papers_text = await self.fetch_arxiv_papers_by_topic(distilled_query, date_info)
            
            if "No papers found" in papers_text:
                logger.info(f"No papers found for query: {distilled_query}")
                del active_searches[user_id]
                return papers_text
            
            # Summarize papers
            await message.channel.send("Found some papers! Generating a summary...")
            summary = await self.summarize_papers(papers_text, distilled_query)
            
            # Store completed search results for this thread
            completed_searches[channel_id] = {
                'papers': papers_text,
                'query': distilled_query,
                'summary': summary,
                'date_info': date_info,
                'raw_query': combined_raw_query
            }
            logger.info(f"Stored search results for thread {channel_id}")
            
            # Mark search as complete
            del active_searches[user_id]
            logger.info(f"Completed search for user {user_id}")
            
            return summary
    
    async def make_thread_name(self, message):
        """Extract a title for the thread from the user's request"""
        logger.info(f"Creating thread name for message: {message.content[:50]}...")
        
        try:
            messages = [
                ChatMessage(role="system", content="Create a short, descriptive title (max 5 words) based on the user's research query. Make sure your response is ONLY the title, no quotes or explanations."),
                ChatMessage(role="user", content=message.content)
            ]
            
            logger.debug(f"Requesting thread name from Mistral")
            
            response = self.client.chat(
                model=MISTRAL_MODEL,
                messages=messages
            )
            
            thread_name = response.choices[0].message.content.strip()
            logger.info(f"Generated thread name: {thread_name}")
            
            # Ensure the thread name is valid (1-100 characters)
            if not thread_name or len(thread_name) > 100:
                # If invalid, create a fallback title
                words = message.content.split()
                thread_name = " ".join(words[:5]) if len(words) > 5 else message.content
                logger.info(f"Using fallback thread name: {thread_name}")
                
            # Final safety check
            if len(thread_name) > 100:
                thread_name = thread_name[:97] + "..."
                logger.info(f"Truncated thread name: {thread_name}")
            elif not thread_name:
                thread_name = "Research Thread"
                logger.info(f"Using default thread name: {thread_name}")
                
            return thread_name
            
        except Exception as e:
            logger.error(f"Error creating thread name: {str(e)}", exc_info=True)
            return "Research Thread"  # Default fallback
