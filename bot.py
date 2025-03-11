import os
import discord
import logging

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent
from user_profiles import get_user_profile, InteractionType

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()


# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")


@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return
    
    # Check if the message is in a profile setup thread
    if isinstance(message.channel, discord.Thread) and message.channel.name.startswith("Profile Setup -"):
        # Skip processing for messages in profile setup threads
        # These will be handled by the create_profile function
        return
    
    # Process User
    user = await get_user_profile(message.author, message, bot)
    if not user:
        return

    original_content = message.content
    # Add user history to the message content for context
    if user and hasattr(user, 'context_string'):
        # Create a new message object with context included
        # message_with_context = discord.Object(id=message.id)
        message.content = f"User History Context:{user.context_string}\n\nCurrent Message:{original_content}"
        # message_with_context.author = message.author
        # message_with_context.channel = message.channel
        
        # # Use the enhanced message for processing
        # message = message_with_context
    
    # Add user message to history (approximate token count - 4 chars per token)
    user.update_history(
        interaction_type=InteractionType.USER,
        text=original_content,
        num_tokens=len(original_content) // 4
    )
    
    # If message is already in a thread, process directly without creating new thread
    if isinstance(message.channel, discord.Thread):
        response = await agent.run(message)
        if response:
            # Add agent response to history
            user.update_history(
                interaction_type=InteractionType.SYSTEM,
                text=response,
                num_tokens=len(response) // 4
            )
            
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.channel.send(chunk)
        return

    # Process the message with the agent
    logger.info(f"Processing message from {message.author}: {message.content}")

    response = await agent.run(message)
    
    # Only create thread and reply if there's a response
    if response:
        # Add agent response to history
        user.update_history(
            interaction_type=InteractionType.SYSTEM,
            text=response,
            num_tokens=len(response) // 4
        )
        
        # Make thread name
        if "profile" in response.lower():
            thread_name = "Profile Setup - " + message.author.name
        else:
            thread_name = await agent.make_thread_name(message)
        
        # Look for existing thread with same name
        existing_thread = None
        for thread in message.channel.threads:
            if thread.name == thread_name:
                existing_thread = thread
                break
        
        # Use existing thread or create new one
        if existing_thread:
            thread = existing_thread
            await message.reply(f"Let's discuss more in the previous thread {thread.mention}")
        # else:
        #     thread = await message.create_thread(name=thread_name)
        
        # Split response into chunks and send in thread
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await thread.send(chunk)


# Commands


# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")


@bot.command(name="arxiv", help="Fetches and summarizes recent arxiv papers")
async def arxiv_command(ctx):
    # Send initial response
    initial_msg = await ctx.send("Fetching recent arxiv papers...")
    
    # Get papers and summary from agent
    response = await agent.run(ctx.message)
    
    # Split response into chunks of 1900 characters (leaving room for formatting)
    chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
    
    # Edit the first message with the first chunk
    await initial_msg.edit(content=chunks[0])
    
    # Send additional chunks as new messages if needed
    for chunk in chunks[1:]:
        await ctx.send(chunk)


# Start the bot, connecting it to the gateway
bot.run(token)
