import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# This will print EVERY model your key can see
for m in client.models.list():
    print(f"AVAILABLE MODEL: {m.name}")