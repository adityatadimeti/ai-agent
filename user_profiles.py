import os
from mistralai import Mistral
import discord
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# Create the directory
USER_DIR = "users/"
os.makedirs(USER_DIR, exist_ok=True)

class User:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.data_path = os.path.join(USER_DIR, user_id)

def get_user_profile(user, message):
    # Convert Discord Member object to string ID
    user_id = str(user.id)
    user_data = os.path.join(USER_DIR, user_id)
    if os.path.exists(user_data):
        return User(user_id)
    else:
        return create_profile(user_id, message)

async def create_profile(user_id, message):
    """Creates a profile directory and a Discord thread for setup."""
    user_data = os.path.join(USER_DIR, user_id)
    os.makedirs(user_data, exist_ok=True)

    # thread_name = f"Profile Setup - {user_id}"
    # thread = await message.create_thread(name=thread_name)

    # message.reply(
    # """
    # Hi! I am an Arxiv bot and will try my best to help you with questions regarding:
    # 1. Research questions
    # 2. Academic literature queries
    # 3. General Q&A and Recommendations with responses sourced from Arxiv

    # I would love to gather a user profile for you so I can better understand your interests, keep track of our conversations, and recommend things to you if you ask

    # Can you please provide me your name and a little description of your research interests, topics you want to explore, etc.?
    # """
    # )

    # pass
    # return thread

    # Return a User object with the created profile
    return User(user_id)
