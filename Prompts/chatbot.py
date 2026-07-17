from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY")
)

chat_history = [
    SystemMessage(content="You are a helpful assistant that summarizes research papers in short detail."),
]

while True:
    user_input = input("\nYou: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    # print(chat_history)
    print("\nAssistant:", result.content)
