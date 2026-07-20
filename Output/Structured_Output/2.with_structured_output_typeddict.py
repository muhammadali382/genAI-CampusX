from typing import TypedDict, Annotated, Optional, Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY")
)

class Review(TypedDict):
    key_themes: Annotated[list[str], 'Write down all the key themes diescussed in the review.']
    summary: Annotated[str, 'A brief summary of the review.']
    sentiment: Annotated[Literal['pos', 'neg'], 'The overall sentiment of the review.']
    pros: Annotated[Optional[list[str]], 'Write down all the pros inside a list.']
    cons: Annotated[Optional[list[str]], 'Write down all the cons inside a list.']
    name: Annotated[Optional[str], 'Write name of the reviewer.']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""TThe LangChain Crash Course is a practical and well-structured learning resource for anyone looking to build applications with large language models. It introduces core concepts such as prompt templates, chains, retrieval-augmented generation (RAG), and agents through clear explanations and hands-on examples. The progression feels natural, making it easy for beginners to follow while still providing useful insights for developers with some prior experience.
Pros:
Clear and beginner-friendly explanations
Hands-on, project-based approach
Covers modern LangChain concepts and workflows
Logical progression from fundamentals to practical applications
Real-world examples that reinforce learning

Cons:
Advanced agent architectures are only briefly covered
Limited discussion of production deployment and scaling
Could benefit from more challenging exercises and larger end-to-end projects
Review by Chat gpt.""")

print(result)
print(result['summary'])
print(result['sentiment'])
print(result.keys())