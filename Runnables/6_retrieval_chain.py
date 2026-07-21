from dotenv import load_dotenv
import os

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Load document
loader = TextLoader("./Runnables/docs.txt")
documents = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# Create retriever
retriever = vectorstore.as_retriever()

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
)

model = ChatHuggingFace(llm=llm)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """
Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}
"""
)

# Convert documents to text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create retrieval chain
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Ask a question
query = "What are the key takeaways from the document?"

# Generate answer
response = chain.invoke(query)

print(response)
