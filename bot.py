import os
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent, active_searches

PREFIX = "!"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord")

# Load environment variables
load_dotenv()

# Create bot with all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Create Mistral agent
agent = MistralAgent()

# Get token from environment variables
token = os.getenv("DISCORD_TOKEN")

@bot.event
async def on_ready():
    """Called when the bot connects to Discord"""
    logger.info(f"{bot.user} has connected to Discord!")

@bot.event
async def on_message(message: discord.Message):
    """Called when a message is sent in any channel the bot can see"""
    # Process commands first
    await bot.process_commands(message)

    # Ignore messages from self or other bots
    if message.author.bot or message.content.startswith(PREFIX):
        return
    
    user_id = message.author.id
    
    # Check if this is a follow-up to a clarification request
    if isinstance(message.channel, discord.Thread) and user_id in active_searches and active_searches[user_id].get('status') == 'awaiting_clarification':
        original_query = active_searches[user_id]['original_query_raw']
        
        # Send initial response
        initial_msg = await message.channel.send("Processing your response...")
        
        # Process the clarification response
        response = await agent.run(message, is_follow_up=True, original_query=original_query)
        
        # Split response at markdown headings and then into chunks if needed
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
            if current_length + line_length > 1900:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(line)
            current_length += line_length
        
        # Add any remaining content as final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Edit the first message with the first chunk
        await initial_msg.edit(content=chunks[0])
        
        # Send additional chunks as new messages if needed
        for chunk in chunks[1:]:
            await message.channel.send(chunk)
        
        return
    
    # If the message is already in a thread, process it
    if isinstance(message.channel, discord.Thread):
        # Check if user has an active search
        if user_id in active_searches:
            await message.channel.send("I'm still processing your previous request. Please wait a moment.")
            return
        
        # Process new message in existing thread
        response = await agent.run(message)
        
        if response:
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.channel.send(chunk)
        return
    
    # Process the message with the agent for new threads
    logger.info(f"Processing message from {message.author}: {message.content}")
    response = await agent.run(message)
    
    # Only create thread and reply if there's a response
    if response:
        # Create thread name
        thread_name = await agent.make_thread_name(message)
        
        # Create a new thread
        try:
            thread = await message.create_thread(name=thread_name if thread_name else "Research Thread")
            
            # Send response in thread
            await thread.send(response)
        except discord.errors.HTTPException as e:
            logger.error(f"Failed to create thread: {e}")
            await message.channel.send(response)

@bot.command(name="arxiv", help="Search for recent papers on arXiv")
async def arxiv_command(ctx, *, query=None):
    """Command to search for recent arXiv papers"""
    if not query:
        await ctx.send("Please provide a search query. Example: `!arxiv quantum computing`")
        return
    
    user_id = ctx.author.id
    
    # Check if user has an active search
    if user_id in active_searches:
        await ctx.send("You already have an active search. Please wait for it to complete.")
        return
    
    # Start a new search
    active_searches[user_id] = {'status': 'awaiting_clarification', 'original_query_raw': query}
    
    # Get clarification
    clarification = await agent.clarify_query(query)
    
    # Create a thread for this search
    thread_name = await agent.make_thread_name(ctx.message)
    try:
        thread = await ctx.message.create_thread(name=thread_name if thread_name else "Research Thread")
        
        # Send clarification request
        await thread.send(f"{clarification}\n\n(Please reply directly to this thread with your clarification)")
    except discord.errors.HTTPException as e:
        logger.error(f"Failed to create thread: {e}")
        await ctx.send(f"{clarification}\n\n(Please reply to this message with your clarification)")

@bot.command(name="status", help="Check the status of your current search")
async def status_command(ctx):
    """Command to check search status"""
    user_id = ctx.author.id
    
    if user_id in active_searches:
        status = active_searches[user_id].get('status', 'in progress')
        await ctx.send(f"Your search is currently {status}.")
    else:
        await ctx.send("You don't have any active searches.")

# Start the bot
bot.run(token)