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
    if message.author == bot.user:
        return

    # Check if this is the first message in the thread
    is_first_message = True
    if isinstance(message.channel, discord.Thread):
        # Get the first message in the thread
        first_message = None
        async for msg in message.channel.history(limit=1, oldest_first=True):
            first_message = msg
            break
        
        # If this message is not the first message, it's a follow-up
        if first_message and first_message.id != message.id:
            is_first_message = False
    
    print(f"Message in thread: {isinstance(message.channel, discord.Thread)}")
    print(f"Is first message: {is_first_message}")
    
    try:
        response = await agent.run(message, is_first_message)
        await message.channel.send(response)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await message.channel.send("I encountered an error processing your request.")


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
