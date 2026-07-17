from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=1, dimension=32)

vector = embedding.embed_query("What is the Capital of Pakistan?")

print(vector)
print(str(vector))
