from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from dotenv import load_dotenv
import os

load_dotenv()


def word_count(text):
    return len(text.split())


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY"),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(
    prompt,
    model,
    parser,
)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count), # lambda x: len(x.split())
    }
)

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain,
)

result = final_chain.invoke({"topic": "AI"})

final_result = f"""{result['joke']}

Word Count: {result['word_count']}"""

print(final_result)
