from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(
            prompt2,
            model,
            parser,
        ),
    }
)

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain,
)

result = final_chain.invoke({"topic": "cricket"})

print("Joke:\n")
print(result["joke"])

print("\n" + "=" * 50 + "\n")

print("Explanation:\n")
print(result["explanation"])
