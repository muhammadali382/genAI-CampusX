from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import load_prompt

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY")
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ]
)

template = load_prompt("./Prompts/template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "paper": paper_input, 
        "style": style_input, 
        "length": length_input
    })
    
    st.write(result.content)
