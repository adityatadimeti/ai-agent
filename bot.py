import os
import discord
import logging
import asyncio
from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent
from user_profiles import get_user_profile, InteractionType
from asyncio import Queue, create_task, gather, Lock, TimeoutError
from collections import deque

PREFIX = "!"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord")

# Load environment variables
load_dotenv()

# Create the bot with all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")

# Constants for scaling
MAX_WORKERS = 10
RATE_LIMIT_DELAY = 0.5  # seconds between messages to avoid rate limits
HEARTBEAT_INTERVAL = 60  # seconds between worker pool health checks

# Message queues
message_queue = Queue()
outbound_message_queue = Queue()

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Spawns worker tasks and message sender.
    """
    logger.info(f"{bot.user} has connected to Discord!")
    
    # Start the worker pool
    for i in range(MAX_WORKERS):
        create_task(worker(i))
    
    # Start the outbound message handler (rate-limited)
    create_task(outbound_message_sender())
    
    # Start the health monitor
    create_task(monitor_worker_health())

@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.
    """
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops
    if message.author.bot or message.content.startswith("!"):
        return
    
    # Check if the message is in a profile setup thread
    if isinstance(message.channel, discord.Thread) and message.channel.name.startswith("Profile Setup -"):
        # Skip processing for messages in profile setup threads
        return

    # Add task to worker pool
    await process_message(message)

@bot.event
async def on_error(event, *args, **kwargs):
    """
    Log errors to prevent silent failures
    """
    logger.error(f"Error in event {event}: {args}")
    import traceback
    logger.error(traceback.format_exc())

async def process_message(message):
    """Enqueue a message for processing"""
    await message_queue.put(message)
    logger.debug(f"Queued message from {message.author} (queue size: {message_queue.qsize()})")

async def monitor_worker_health():
    """Periodically check worker health and restart if needed"""
    while True:
        logger.info(f"Health check - Queue size: {message_queue.qsize()}, Workers: {MAX_WORKERS}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)

async def worker(worker_id):
    """Worker task to process incoming messages"""
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            message = await message_queue.get()
            logger.debug(f"Worker {worker_id} processing message from {message.author}")
            
            try:
                # Process User
                user = await get_user_profile(message.author, message, bot)
                if not user:
                    continue

                original_content = message.content
                if user and hasattr(user, 'context_string'):
                    message.content = f"User History Context:{user.context_string}\n\nCurrent Message:{original_content}"
                
                # Add user message to history
                user.update_history(
                    interaction_type=InteractionType.USER,
                    text=original_content,
                    num_tokens=len(original_content) // 4
                )
                
                # If message is already in a thread, process directly
                if isinstance(message.channel, discord.Thread):
                    response = await agent.run(message)
                    if response:
                        user.update_history(
                            interaction_type=InteractionType.SYSTEM,
                            text=response,
                            num_tokens=len(response) // 4
                        )
                        
                        await queue_chunked_message(message, response)
                    continue

                # Process the message with the agent
                logger.info(f"Processing message from {message.author}")

                # Run the scheduler pipeline
                response = await agent.run(message)
                
                if response:
                    user.update_history(
                        interaction_type=InteractionType.SYSTEM,
                        text=response,
                        num_tokens=len(response) // 4
                    )
                    
                    # Create a thread if needed, using create_task to not block
                    thread_name = "Profile Setup - " + message.author.name if "profile" in response.lower() else await agent.make_thread_name(message)
                    
                    # Check if the thread already exists - do this asynchronously
                    existing_thread = await find_existing_thread(message.channel, thread_name)
                    
                    # If the thread already exists, use it
                    if existing_thread:
                        thread = existing_thread
                        await queue_message(message.channel, f"Let's discuss more in the previous thread {thread.mention}")
                    
                    # Send response in chunks using the rate-limited queue
                    await queue_chunked_message(message, response)

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
            finally:
                message_queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered error: {str(e)}")
            # Sleep briefly to avoid tight error loops
            await asyncio.sleep(1)

async def find_existing_thread(channel, thread_name):
    """Find an existing thread by name, with timeout protection"""
    try:
        async for thread in channel.archived_threads(limit=100):
            if thread.name == thread_name:
                return thread
                
        for thread in channel.threads:
            if thread.name == thread_name:
                return thread
                
        return None
    except Exception as e:
        logger.error(f"Error finding thread: {str(e)}")
        return None

async def queue_message(channel, content):
    """Queue a message to be sent with rate limiting"""
    await outbound_message_queue.put((channel, content))

async def queue_chunked_message(message, response):
    """Queue chunked messages with rate limiting"""
    chunks = split_response_into_chunks(response)
    
    for chunk in chunks:
        await outbound_message_queue.put((message.channel, chunk))

def split_response_into_chunks(response, max_length=1900):
    """Split response into chunks, respecting markdown and length limits"""
    chunks = []
    sections = response.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in sections:
        # If line starts with heading (#) and we have content, start new chunk
        if line.lstrip().startswith('#') and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        # Add line to current chunk
        line_length = len(line) + 1  # +1 for newline
        if current_length + line_length > max_length:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(line)
        current_length += line_length
    
    # Add any remaining content as final chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        
    return chunks

async def outbound_message_sender():
    """Worker to send outbound messages with rate limiting"""
    logger.info("Started outbound message sender")
    
    while True:
        try:
            channel, content = await outbound_message_queue.get()
            
            try:
                await channel.send(content)
                logger.debug(f"Sent message to {channel.name} (queue size: {outbound_message_queue.qsize()})")
            except discord.HTTPException as e:
                logger.error(f"Failed to send message: {str(e)}")
                # If we hit a rate limit, requeue with a delay
                if e.code == 429:
                    logger.warning(f"Rate limited! Retrying in {e.retry_after} seconds")
                    await asyncio.sleep(e.retry_after)
                    await outbound_message_queue.put((channel, content))
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
            
            # Rate limit ourselves to avoid hitting Discord limits
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            logger.error(f"Error in outbound message sender: {str(e)}")
            await asyncio.sleep(1)
        finally:
            outbound_message_queue.task_done()

# Start the bot, connecting it to the gateway
if __name__ == "__main__":
    try:
        bot.run(token)
    except Exception as e:
        logger.critical(f"Failed to start bot: {str(e)}")
