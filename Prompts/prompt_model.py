from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
    temperature=0.2,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    response = model.invoke(prompt)

    print("\nAssistant:", response.content)


# result =model.invoke("you only have old data or real time data?")

# print(result.content)