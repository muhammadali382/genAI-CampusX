from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=1, dimension=32)

documents = [
    "Islamabad is the capital of Pakistan",
    "Kolkata is the capital of West Bengal.",
]

vector = embedding.embed_documents(documents)

print(vector)
print(str(vector))
2