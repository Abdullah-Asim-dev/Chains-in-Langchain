import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Environment variables load karein
load_dotenv()

# 2. Prompt Template setup
prompt = PromptTemplate(
    template="Generate 5 interesting facts about \n {topic}"
)

# 3. Bilkul Naya aur Active Groq Model lagayein
model = ChatGroq(
    model='openai/gpt-oss-20b',  # Yeh model abhi active aur working hai
    temperature=0.1
)

parser = StrOutputParser()

# 4. Chain banayein aur run karein
chain = prompt | model | parser
result = chain.invoke({"topic": "cricket"})

print(result)
# chain ko visualize kar sakte hoo
# chain.get-graph().print_ascii()
