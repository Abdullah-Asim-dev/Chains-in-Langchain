# 🧬 LangChain Execution Layers (LCEL) Masterclass

[![LangChain Version](https://shields.io)](https://langchain.com)
[![Python Version](https://shields.io)](https://python.org)
[![Provider](https://shields.io)](https://groq.com)
[![License](https://shields.io)](LICENSE)

A production-ready implementation of complex design patterns using **LangChain Expression Language (LCEL)**. Inspired by the foundational principles of the **CampusX (Nitish Singh)** Generative AI curriculum, this repository transitions abstract agentic workflows into modular, scalable, and deterministic Python components.

---

## 📌 Table of Contents
1. [Core Features](#-core-features)
2. [Why LCEL? (Abstracting Manual Pipelines)](#-why-lcel-abstracting-manual-pipelines)
3. [Repository Directory Layout](#-repository-directory-layout)
4. [Installation & Local Setup](#%EF%B8%8F-installation--local-setup)
5. [Architectural Blueprints & Implementations](#-architectural-blueprints--implementations)
    - [Simple Chain (The Atomic Unit)](#1-simple-chain-the-atomic-unit)
    - [Sequential Chain (Linear State Transitions)](#2-sequential-chain-linear-state-transitions)
    - [Parallel Chain (Concurrently Partitioned RAG)](#3-parallel-chain-concurrently-partitioned-rag)
    - [Conditional Chain (Dynamic Pydantic Routing)](#4-conditional-chain-dynamic-pydantic-routing)
6. [Pipeline Graph Visualisation](#-pipeline-graph-visualisation)
7. [Production Operational Guardrails](#-production-operational-guardrails)
8. [Acknowledgments & References](#-acknowledgments--references)

---

## 🚀 Core Features
* **State-of-the-Art Model Routing:** Fully patched for modern 2026 inference endpoints (`openai/gpt-oss-20b`, `qwen/qwen3.6-27b`).
* **Deterministic Parsing:** Strict validation using `PydanticOutputParser` to completely avoid model hallucinations in routing logic.
* **Parallel Processing:** Independent sub-graph evaluations using standard `RunnableParallel` primitives to reduce network latency.
* **Telemetry Ready:** Built-in hooks for parsing directed acyclic graphs (DAGs) using native Mermaid render layers.

---

## 🧠 Why LCEL? (Abstracting Manual Pipelines)
In traditional LLM application structures, managing state flow is highly imperative, error-prone, and heavily manual:
1. Interpolating state variable payloads into a template.
2. Handling synchronous network invocations (`llm.invoke`).
3. Manually extracting raw substrings while catching parser errors.

**LCEL transitions this into a declarative paradigm using the Pipe Operator (`|`):**
The underlying runtime natively converts your functional code blocks into optimized execution pipelines, ensuring that the **output signature** of step $N$ seamlessly binds to the **input signature** of step $N+1$.

---

## 📁 Repository Directory Layout
```text
├── .env.example               # Template file for secure API credentials
├── README.md                  # Comprehensive technical documentation
├── requirements.txt           # Standard Python package pinning configurations
└── workflows/                 # Complete source code modules
    ├── __init__.py
    ├── simple_pipeline.py     # Single-step inference engine
    ├── sequential_flow.py     # Linear cascading state transformations
    ├── parallel_rag.py        # Concurrent asset generation layout
    └── conditional_router.py  # Intent-driven conditional branch engine
```

---

## 🛠️ Installation & Local Setup

### 1. Provision Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Packages & Update Dependencies
```bash
pip install --upgrade pip
pip install -U langchain-groq langchain-core python-dotenv pydantic
```

### 3. Setup Runtime Variables
Create a `.env` file at the root level of your project workspace:
```env
GROQ_API_KEY=gsk_your_validated_production_key_here
```
> **Security Guardrail:** Never wrap your keys in double quotes `""` or single quotes `''` inside the `.env` file to prevent string pollution issues during initialization.

---

## 🧬 Architectural Blueprints & Implementations

### 1. Simple Chain (The Atomic Unit)
Links a basic user state payload to an inference pipeline utilizing string stream standardizers.

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 distinct interesting facts regarding the following topic:\n{topic}",
    input_variables=["topic"]
)

# Initialising the modern platform-native model
model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.1)
parser = StrOutputParser()

# Declarative assembly via LCEL
simple_chain = prompt | model | parser
print(simple_chain.invoke({"topic": "Cricket"}))
```

---

### 2. Sequential Chain (Linear State Transitions)
Useful for hierarchical multi-prompt pipelines where step $B$ requires structural alignment with step $A$'s outputs.

```text
[Input State] ──> PromptTemplate_A ──> LLM ──> StrParser ──> [Intermediate Report] ──> PromptTemplate_B ──> LLM ──> StrParser ──> [Final Summary Output]
```

```python
prompt_report = PromptTemplate.from_template("Generate a comprehensive technological report on {topic}")
prompt_summary = PromptTemplate.from_template("Extract a 5-pointer executive summary from this body of text:\n{text}")

model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.1)
parser = StrOutputParser()

# Continuous cascading stream orchestration
sequential_pipeline = prompt_report | model | parser | prompt_summary | model | parser
print(sequential_pipeline.invoke({"topic": "Artificial Intelligence Frameworks"}))
```

---

### 3. Parallel Chain (Concurrently Partitioned RAG)
Optimizes system performance by broadcasting a singular input text across multiple concurrent evaluation threads.

```python
from langchain_core.runnables import RunnableParallel

notes_prompt = PromptTemplate.from_template("Extract academic study notes from this transcript:\n{text}")
quiz_prompt = PromptTemplate.from_template("Formulate a 5-question multi-choice quiz from this material:\n{text}")
merge_prompt = PromptTemplate.from_template("Compile the following data assets into a cohesive technical document:\nNotes:\n{notes}\n\nQuiz:\n{quiz}")

model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.1)
parser = StrOutputParser()

# Mapping concurrent execution paths
async_workers = RunnableParallel(
    notes=notes_prompt | model | parser,
    quiz=quiz_prompt | model | parser
)

# Feeding concurrent outputs back into a sequential compiler
orchestration_graph = async_workers | merge_prompt | model | parser
```

---

### 4. Conditional Chain (Dynamic Pydantic Routing)
An enterprise-grade routing schema that leverages strongly-typed schemas to handle complex business logic. It reads runtime contexts and routes execution paths safely and dynamically.

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Structuring evaluation matrices via Pydantic
class SentimentEvaluation(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Extracted classification indicator')

pydantic_parser = PydanticOutputParser(pydantic_object=SentimentEvaluation)
str_parser = StrOutputParser()

classifier_prompt = PromptTemplate(
    template="Evaluate the sentiment classification profile of this feedback text.\n{feedback}\n\n{format_instructions}",
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)
classification_node = classifier_prompt | model | pydantic_parser

# Routing vectors
positive_handler = PromptTemplate.from_template("Draft an appreciative corporate thank-you response for:\n{feedback}")
negative_handler = PromptTemplate.from_template("Draft an empathetic customer-success mitigation dispatch for:\n{feedback}")

conditional_branch = RunnableBranch(
    (lambda x: x['evaluation'].sentiment == 'positive', (lambda x: {'feedback': x['feedback']}) | positive_handler | model | str_parser),
    (lambda x: x['evaluation'].sentiment == 'negative', (lambda x: {'feedback': x['feedback']}) | negative_handler | model | str_parser),
    RunnableLambda(lambda x: 'Fallthrough: Structural invariant encountered.')
)

# Wrapping pipeline to prevent input payload deletion
complete_pipeline = RunnableParallel(
    evaluation=classification_node,
    feedback=lambda x: x['feedback']
)

execution_graph = complete_pipeline | conditional_branch
print(execution_graph.invoke({'feedback': 'The performance of this smartphone is completely abysmal.'}))
```

---

## 📊 Pipeline Graph Visualisation

Since raw ASCII charts can drop nested branch layers when complex `RunnableBranch` or custom lambdas are inside the loop, the recommended approach is to generate a deterministic **Mermaid Flowchart Code**:

```python
# Insert at the end of your pipeline scripts to capture the execution graph
print(execution_graph.get_graph().draw_mermaid())
```

### Steps to Render:
v
