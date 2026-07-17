from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model_name="claude-opus-4-8", 
    temperature=0.5, 
    max_completions_tokens=10
)

result = model.invoke("Write a poem about the ocean.")

print(result.content)
