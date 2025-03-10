import os
from mistralai import Mistral
import discord
from discord.ext import commands
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
import json
from enum import Enum

# Create the directory
USER_DIR = "users/"
os.makedirs(USER_DIR, exist_ok=True)

# Possibly, if you want, can run llm over the user profile and save a structured format to .txt file

# def call_user_llm(prompt)
    
class UserDetails(BaseModel):
    name: str = Field(description="Extract the User's Name")
    interests: list[str] = Field(description="Extract a list of things that the user is interested in")
    justification: list[str] = Field(
        None, description="Justification for each to-do item and why it is in its respective position in the list"
    )

class InteractionType(Enum):
    USER = "USER"
    SYSTEM = "SYSTEM"

MAX_MEMORY_TOKENS = 500

class Interaction:
    def __init__(self, interaction_type: InteractionType, text: str, num_tokens: int):
        self.interaction_type = interaction_type
        self.text = text
        self.num_tokens = num_tokens

class User:
    def __init__(self, user_id: str, profile_complete: bool = False) -> None:
        self.user_id = user_id
        self.profile_complete = profile_complete
        self.data_path = os.path.join(USER_DIR, user_id)
        
        # Check if files exist before trying to read them
        profile_path = os.path.join(self.data_path, "profile.txt")
        memory_path = os.path.join(self.data_path, "memory.json")
        
        if os.path.exists(profile_path):
            self.profile = open(profile_path, "r").read()
        else:
            self.profile = ""
            
        if os.path.exists(memory_path):
            self.context_dict = json.load(open(memory_path))
        else:
            self.context_dict = {
                'message_history': [],
                'num_token': 0
            }
            
        self.context_string = self.context_to_string()

    def context_to_string(self):
        context_string = "\nBEGINNING OF CONTEXT WITH USER\n"
        context_string += "(NOTE THAT THIS IS LIMITED CONTEXT!)\n"
        print(self.context_dict)
        for interaction in self.context_dict['message_history']:
            context_string += f"{interaction.interaction_type}: \n {interaction.text} \n\n"

        return context_string
    
    def update_history(self, new_interaction):
        if new_interaction.num_tokens > MAX_MEMORY_TOKENS:
            new_interaction.text = new_interaction.text[-MAX_MEMORY_TOKENS:]
        while self.context_dict['num_token'] + new_interaction.num_tokens > MAX_MEMORY_TOKENS:
            out = self.context_dict['message_history'].pop(0)
            self.context_dict['num_token'] -= out
        self.context_dict['message_history'].append(new_interaction)
        self.context_string = self.context_to_string
        

async def create_profile(user: discord.Member, message: discord.Message, bot: commands.Bot) -> User:
    """
    Creates a profile folder and opens a thread for profile setup.
    """
    user_id = str(user.id)
    user_data = os.path.join(USER_DIR, user_id)
    os.makedirs(user_data, exist_ok=True)

    # Create a dedicated thread for profile setup
    thread = await message.create_thread(name=f"Profile Setup - {user_id}")
    await thread.send(
        """
        Hi! Looks like you are a new user!
        
        I am an Arxiv bot and will try my best to help you with questions regarding:
        1. Research questions
        2. Academic literature queries
        3. General Q&A and Recommendations with responses sourced from Arxiv

        I would love to gather a user profile for you so I can better understand your interests, keep track of our conversations, and recommend things to you if you ask

        Can you please provide me your name and a short description of your research interests, topics you want to explore, etc.?
        """
    )

    def check(m: discord.Message):
        return m.author.id == user.id and m.channel == thread
    
    # llm = ChatMistralAI(model="mistral-small-latest")

    while True:
        try:
            blurb_msg = await bot.wait_for('message', check=check, timeout=120)
            print("Checking blurb message")
            blurb = blurb_msg.content

            await thread.send(
                f"""
                Confirming that this is your preferred description?

                {str(blurb)}

                If so, say "yes". Otherwise, submit a new one
                """
            )
            confirmation_msg = await bot.wait_for('message', check=check, timeout=120)
            confirmation = str(confirmation_msg.content).lower()

            if confirmation != "yes":
                continue
            
            # Save the name (for example, in a file; you can expand this as needed)
            with open(os.path.join(user_data, "profile.txt"), "w") as f:
                f.write(blurb)
                
            memory_data = {   
                'message_history': [],
                'num_token': 0
            }
            with open(os.path.join(user_data, "memory.json"), "w") as f:
                json.dump(memory_data, f)

            await thread.send(f"Thanks! Your profile is now set up. Now going back to your query!")
            
            # Return a user object indicating the profile is complete
            return User(user_id, profile_complete=True)
            
        except Exception as e:
            await thread.send(
            """
            Profile setup timed out or encountered an error. Please try again later.
            """)
            # You might want to handle cleanup or retries here.
            return None


async def get_user_profile(user: discord.Member, message: discord.Message, bot: commands.Bot) -> User:
    """
    Returns the user profile. If the profile doesn't exist, triggers the profile creation process.
    """
    user_id = str(user.id)
    user_data = os.path.join(USER_DIR, user_id)
    if os.path.exists(user_data):
        # In a real application, you might also check if the profile data is complete.
        return User(user_id, profile_complete=True)
    else:
        # Start the profile creation "tangent" and wait for it to complete.
        return await create_profile(user, message, bot)

