from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-3.5-turbo-instruct", 
    temperature=0.5, 
    max_completions_tokens=10
)

result = model.invoke("Write a poem about the ocean.")
print(result.content)
    