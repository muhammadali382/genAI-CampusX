from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

result = llm.invoke("Write a poem about the ocean.")

print("NOt bad")
print(result)