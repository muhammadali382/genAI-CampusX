from typing import TypedDict, Annotated, Optional, Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pydantic import BaseModel, EmailStr, Field

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY")
)

class Review(BaseModel):
    key_themes: list[str] =Field(description="Write down all the key themes discussed "
    "in the review.")
    summary: str = Field(description="A brief summary of the review.")
    sentiment: Literal["pos", "neg"] =Field(description="The overall sentiment of " \
    "the review.")
    pros:Optional[list[str]] =Field(description="Write down all the pros inside a " \
    "list.")
    cons:Optional[list[str]] =Field(description="Write down all the cons inside a " \
    "list.")
    name: Optional[str] =Field(description="Write the name of the reviewer if " \
    "mentioned.")



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
print(result.name)
print(result.summary)
print(result.sentiment)
print(result.key_themes)