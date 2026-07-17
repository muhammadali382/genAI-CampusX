from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful {domain} expert"),
    ('human', "Explain in simple terms, what is '{paper}' in a {style} manner with a {length} explanation."),
])

prompt = chat_template.invoke({
    "domain": "AI research",
    "paper": "Attention Is All You Need",
    "style": "Beginner-Friendly",
    "length": "Short (1-2 paragraphs)"
})

print(prompt)  