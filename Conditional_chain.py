import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Environment variables load karein
load_dotenv()

# 2. Model setup
model = ChatGroq(
    model='openai/gpt-oss-20b',  # Yeh model abhi active aur working hai
    temperature=0.1
)

# Aapka missing parser jo branch me use ho raha hai
str_parser = StrOutputParser()

# 3. Pydantic Model banayein
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# 4. Pydantic Output Parser banayein (Syntax corrected here)
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

# 5. Prompt Template setup
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text into positive or negative.\n{feedback}\n\n{format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)

# 6. Chains aur Prompts ka setup
classifier_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
    template='Write an appreciative response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an apologetic response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# 7. Router Branch (Aapka original logic, minor syntax fixes ke sath)
branch_chain = RunnableBranch(
    # Pehle condition check hogi, phir lambda ke zariye original feedback text aage pass hoga
    (lambda x: x['sentiment'].sentiment == 'positive', (lambda x: {'feedback': x['feedback']}) | prompt2 | model | str_parser),
    (lambda x: x['sentiment'].sentiment == 'negative', (lambda x: {'feedback': x['feedback']}) | prompt3 | model | str_parser),
    RunnableLambda(lambda x: 'could not find the sentiment')  # Yahan comma missing tha
)   

# 8. Dono ko jodne ke liye original input aur sentiment dono ko ek sath bhejna hoga
main_pipeline = RunnableParallel(
    sentiment=classifier_chain,
    feedback=lambda x: x['feedback']
)

chain = main_pipeline | branch_chain

# 9. Run karein
result = chain.invoke({'feedback': 'this is a terrible smart phone'})
print(result)
# Yeh line crash nahi hogi aur ek diagram code generate karegi
print(chain.get_graph().draw_mermaid())
