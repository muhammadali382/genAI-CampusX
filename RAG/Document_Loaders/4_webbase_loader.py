from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
)

prompt = PromptTemplate(
    template="""
Answer the following question using only the information provided.

Question:
{question}

Context:
{text}
""",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

url = "https://www.youtube.com/watch?v=bL92ALSZ2Cg&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=12"

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

response = chain.invoke(
    {
        "question": "what is discussed in the video?",
        "text": docs[0].page_content,
    }
)

print(response)

