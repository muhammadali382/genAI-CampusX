from langchain_openai import OpenAIEmbedding
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbedding(model = "text-embedding-3-large", dimension=300)

#5 different documents to compare
documents = [
    "Islamabad is the capital of Pakistan.",
    "Kolkata is the capital of West Bengal.",
    "Karachi is the capital of Sindh.",
    "Lahore is the capital of Punjab.",
    "Peshawar is the capital of Khyber Pakhtunkhwa."
]
query = "What is the capital of Punjab?"

doc_vector = embedding.embed_documents(documents)

query_vector = embedding.embed_query(query)

scores = cosine_similarity(query_vector.reshape(1, -1), doc_vector)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(f"Document: {documents[index]}, Score: {score}")
