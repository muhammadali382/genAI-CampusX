import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful assistant that summarizes research papers in short detail."),
    HumanMessage(content="Summarize the research paper 'Attention Is All You Need' in a Beginner-Friendly manner with a Short (1-2 paragraphs) explanation."),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
