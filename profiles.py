import os
from mistralai import Mistral
import discord
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


USER_DIR = "users/"

def create_profile(user):
    # Ensure that the User Directory exists
    os.makedirs(USER_DIR, exist_ok=True)

    # Find the user path and make if new
    user_data = os.path.join(USER_DIR, user)
    os.makedirs(user_data, exist_ok=True)

    # x

