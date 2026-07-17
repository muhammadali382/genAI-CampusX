from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512},
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
)
model = ChatHuggingFace(llm=llm)

print(model.invoke("What is the Capital of Pakistan?").content)