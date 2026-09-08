import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Environment variables load karein
load_dotenv()
prompt1=PromptTemplate(
    template='generate a detailed report on /n {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='generate a five pointers summary on text  /n {text}',
    input_variables=['text']
)
model = ChatGroq(
    model='openai/gpt-oss-20b',  # Yeh model abhi active aur working hai
    temperature=0.1
)
parser=StrOutputParser()
chain=prompt1|model|parser|prompt2|model|parser
result=chain.invoke({'topic':'most unemployment degree in pakistan'})
print(result)
