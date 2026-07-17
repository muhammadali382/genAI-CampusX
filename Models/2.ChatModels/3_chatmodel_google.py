from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from google import genai
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash",
    temperature=0.5,
    max_output_tokens=512,
)

response = model.invoke("What is the Capital of Pakistan?")

print(response)

# List all the models

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# for model in client.models.list():
#     print(model.name)
