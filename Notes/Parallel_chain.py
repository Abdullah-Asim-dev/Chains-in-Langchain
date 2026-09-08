import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# Model 1 setup
model1 = ChatGroq(
    model='openai/gpt-oss-20b',  
    temperature=0.1
)

# FIXED: Direct ChatGroq initialize karein, bina kisi 'llm=llm' wrap ke!
model2 = ChatGroq(
    model='llama-3.1-8b-instant',  
    temperature=0.1
)

# Prompts setup
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# Parallel Chain Execution
parallel_chain = RunnableParallel({
  'notes': prompt1 | model1 | parser,
  'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
final_chain = parallel_chain | merge_chain

text = """Machine learning (ML) represents a transformative frontier within artificial intelligence (AI) that empowers computer systems to autonomously extract knowledge, discern intricate patterns, and make highly accurate predictions directly from data without being explicitly programmed for specific outcomes."""

# Execute Chain
result = final_chain.invoke({'text': text})
print(result)
