# 8. Tracing & Monitoring AI Applications

---

## Table of Contents

1. [What is Tracing & Why Do We Need It?](#1-what-is-tracing--why-do-we-need-it)
2. [Traditional Observability Tools (Not AI-Specific)](#2-traditional-observability-tools-not-ai-specific)
3. [AI-Specific Tracing Tools](#3-ai-specific-tracing-tools)
4. [LangSmith — Deep Dive](#4-langsmith--deep-dive)
5. [Langfuse — Open Source Alternative](#5-langfuse--open-source-alternative)
6. [LangSmith vs Langfuse Comparison](#6-langsmith-vs-langfuse-comparison)

---

## 1. What is Tracing & Why Do We Need It?

### The Problem

When you build an AI application (chatbot, RAG pipeline, agent), **a single user query can trigger dozens of internal steps**:

```
User Query
  → Prompt Template fills in variables
    → LLM Call #1 (e.g., GPT-4)
      → Tool Call (e.g., search API)
        → LLM Call #2 (summarize results)
          → Final Response
```

Without tracing, you have **zero visibility** into:
- Which step failed or returned bad output
- How much each call cost (tokens / dollars)
- How long each step took (latency)
- What the intermediate inputs & outputs were
- Whether the user was satisfied with the final answer

### What is Tracing?

> **Tracing** = Recording the entire execution flow of an AI application — every LLM call, tool invocation, retrieval step, and chain — as a structured tree of **spans** so you can debug, monitor, and improve it.

### Key Concepts

| Term | Meaning |
|------|---------|
| **Trace** | The full journey of a single request from input → output |
| **Span** | One unit of work inside a trace (e.g., one LLM call, one retriever call) |
| **Run** | LangSmith's term for a span |
| **Parent / Child** | Spans are nested — a chain span contains LLM spans inside it |
| **Metadata** | Extra info attached to a span (model name, user ID, etc.) |
| **Feedback** | Human or automated scores (thumbs up/down, correctness) |

### Architecture Diagram — Where Tracing Fits

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR AI APPLICATION                      │
│                                                             │
│  User Query ──► Prompt ──► LLM ──► Tool ──► LLM ──► Reply  │
│       │            │         │        │        │        │    │
│       ▼            ▼         ▼        ▼        ▼        ▼    │
│   ┌─────────────────────────────────────────────────────┐   │
│   │          TRACING SDK (instrumentation)              │   │
│   │    Captures: inputs, outputs, latency, tokens,      │   │
│   │              errors, metadata for EVERY step         │   │
│   └──────────────────────┬──────────────────────────────┘   │
└──────────────────────────┼──────────────────────────────────┘
                           │  sends traces
                           ▼
              ┌────────────────────────┐
              │   TRACING BACKEND      │
              │  (LangSmith / Langfuse)│
              │                        │
              │  • Dashboard           │
              │  • Cost Analytics      │
              │  • Debug Traces        │
              │  • Prompt Management   │
              │  • Feedback Collection │
              │  • Evaluation Runs     │
              └────────────────────────┘
```

---

## 2. Traditional Observability Tools (Not AI-Specific)

> These tools are built for **general software monitoring** (APIs, microservices, infra). They are **NOT designed for LLM/AI tracing** but good to know about.

| Tool | What It Does | AI-Relevant? |
|------|-------------|--------------|
| **OpenTelemetry** | Open standard for traces, metrics, logs across any app | ⚠️ Generic — no LLM-aware spans |
| **Grafana** | Visualization/dashboard tool — displays metrics & logs | ⚠️ Can display AI metrics if you push them |
| **Prometheus** | Time-series metrics DB (CPU, memory, request counts) | ❌ Not for LLM token tracking |
| **Loki** | Log aggregation system (like a searchable log store) | ❌ Raw logs, no trace tree structure |

### Why These Are NOT Enough for AI Apps

```
Traditional Tools                    AI Tracing Tools
─────────────────                    ────────────────
✗ No concept of "LLM call"          ✓ First-class LLM span type
✗ No token counting                 ✓ Auto token + cost tracking
✗ No prompt/completion capture      ✓ Full I/O capture per step
✗ No chain/agent nesting view       ✓ Nested trace tree view
✗ No prompt versioning              ✓ Prompt hub / management
✗ No human feedback system          ✓ Built-in feedback & eval
```

**Bottom Line:** For AI applications, use **LangSmith** or **Langfuse** (covered below). Use Grafana/Prometheus only for infra-level monitoring alongside them.

---

## 3. AI-Specific Tracing Tools

| Tool | Open Source? | Best For |
|------|-------------|----------|
| **LangSmith** | ❌ No (proprietary, by LangChain) | Teams already using LangChain, enterprise |
| **Langfuse** | ✅ Yes (self-hostable) | Teams wanting full control, open source |
| **Phoenix (Arize)** | ✅ Yes | ML-focused teams, embeddings analysis |
| **Helicone** | Partial | Simple proxy-based logging |
| **Weights & Biases (W&B)** | Partial | ML experiment tracking + LLM |

> We'll deep dive into **LangSmith** (industry standard) and **Langfuse** (best open-source alternative).

---

## 4. LangSmith — Deep Dive

### 4.1 What is LangSmith?

> **LangSmith** is a **proprietary platform by LangChain Inc.** for tracing, debugging, testing, evaluating, and monitoring LLM applications. It works with **any LLM framework** (not just LangChain).

```
┌──────────────────────────────────────────────────┐
│                  LANGSMITH PLATFORM              │
│                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  TRACING   │  │  TESTING & │  │  PROMPT    │ │
│  │  & DEBUG   │  │  EVALUATION│  │  HUB       │ │
│  │            │  │            │  │            │ │
│  │ • Trace    │  │ • Datasets │  │ • Version  │ │
│  │   trees    │  │ • Auto     │  │   control  │ │
│  │ • I/O      │  │   evals    │  │ • Pull in  │ │
│  │   capture  │  │ • Compare  │  │   code     │ │
│  │ • Latency  │  │   runs     │  │ • Playground│ │
│  │ • Errors   │  │ • Scoring  │  │ • Share    │ │
│  └────────────┘  └────────────┘  └────────────┘ │
│                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ MONITORING │  │ FEEDBACK   │  │ DASHBOARD  │ │
│  │            │  │            │  │            │ │
│  │ • Cost/    │  │ • Thumbs   │  │ • Cost     │ │
│  │   tokens   │  │   up/down  │  │   graphs   │ │
│  │ • Latency  │  │ • Scores   │  │ • Usage    │ │
│  │   p50/p99  │  │ • Comments │  │   metrics  │ │
│  │ • Error    │  │ • Auto     │  │ • Model    │ │
│  │   rates    │  │   feedback │  │   breakdown│ │
│  └────────────┘  └────────────┘  └────────────┘ │
└──────────────────────────────────────────────────┘
```

---

### 4.2 Installation & Setup

#### Step 1 — Install the SDK

```bash
pip install langsmith langchain langchain-openai
```

#### Step 2 — Create a LangSmith Account

1. Go to **https://smith.langchain.com**
2. Sign up (free tier available)
3. Go to **Settings → API Keys → Create API Key**
4. Copy the key

#### Step 3 — Set Environment Variables

```bash
# Required
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls__xxxxxxxxxxxxxxxx"

# Optional but recommended
export LANGCHAIN_PROJECT="my-ai-project"       # organizes traces into projects
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"  # default

# Your LLM API key
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

**Windows PowerShell:**
```powershell
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_API_KEY = "ls__xxxxxxxxxxxxxxxx"
$env:LANGCHAIN_PROJECT = "my-ai-project"
$env:OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
```

**Using `.env` file (recommended):**
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=my-ai-project
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

```python
from dotenv import load_dotenv
load_dotenv()
```

> **That's it!** Once these env vars are set, **every LangChain call is automatically traced** — zero code changes needed.

---

### 4.3 How to Monitor via LangSmith — Code Examples

#### Basic Tracing (Automatic with LangChain)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Traces are automatically sent to LangSmith
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm

# This call is automatically traced!
response = chain.invoke({"question": "What is LangSmith?"})
print(response.content)
```

> After running this, go to **smith.langchain.com** → Your project → You'll see the full trace tree!

#### Manual Tracing with `@traceable` Decorator (Non-LangChain Code)

```python
from langsmith import traceable
import openai

client = openai.OpenAI()

@traceable(
    name="My Custom LLM Call",      # name shown in LangSmith
    run_type="llm",                  # type: llm | chain | tool | retriever
    metadata={"version": "1.0"}     # custom metadata
)
def call_gpt(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

@traceable(name="Full Pipeline", run_type="chain")
def my_pipeline(user_input: str) -> str:
    # This becomes a child span under "Full Pipeline"
    answer = call_gpt(user_input)
    return answer

# Run it — traces appear in LangSmith!
result = my_pipeline("Explain quantum computing")
print(result)
```

#### Tracing a RAG Pipeline

```python
from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

@traceable(name="retrieve_docs", run_type="retriever")
def retrieve(query: str, vectorstore):
    docs = vectorstore.similarity_search(query, k=3)
    return docs

@traceable(name="generate_answer", run_type="chain")
def rag_pipeline(question: str, vectorstore):
    # Step 1: Retrieve — shows as child span
    docs = retrieve(question, vectorstore)
    context = "\n".join([d.page_content for d in docs])
    
    # Step 2: Generate
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content
```

```
Trace Tree in LangSmith:
─────────────────────────
📦 rag_pipeline (chain) ──── 2.3s total
  ├── 🔍 retrieve_docs (retriever) ──── 0.4s
  │     └── input: "What is RAG?"
  │     └── output: [Doc1, Doc2, Doc3]
  ├── 📝 ChatPromptTemplate ──── 0.001s
  └── 🤖 ChatOpenAI (llm) ──── 1.9s
        ├── input tokens: 450
        ├── output tokens: 120  
        ├── cost: $0.002
        └── output: "RAG stands for..."
```

---

### 4.4 Free vs Paid — LangSmith Pricing

| Feature | **Free (Developer)** | **Plus ($39/seat/mo)** | **Enterprise (Custom)** |
|---------|---------------------|----------------------|------------------------|
| Traces | **5,000 traces/mo** | **Unlimited** | **Unlimited** |
| Data Retention | 14 days | 400 days | Custom |
| Team Members | 1 | Up to 10 | Unlimited |
| Projects | Limited | Unlimited | Unlimited |
| Prompt Hub | ✅ Basic | ✅ Full | ✅ Full |
| Evaluations | ✅ Basic | ✅ Advanced | ✅ Advanced |
| RBAC (Role Access) | ❌ | ❌ | ✅ |
| SSO / SAML | ❌ | ❌ | ✅ |
| Self-Hosting | ❌ | ❌ | ✅ |
| SLA / Support | ❌ Community | Email | Dedicated |
| Annotations Queue | ❌ | ✅ | ✅ |
| Online Evaluations | ❌ | ✅ | ✅ |

> **Key Limitation:** LangSmith is **NOT open source**. You **cannot self-host** on the free/plus tier. Your trace data lives on LangChain's servers. For companies with data privacy concerns, consider **Langfuse** (Section 5).

---

### 4.5 Complete Theory of Tracing

#### How Tracing Works Under the Hood

```
┌──────────────────────────────────────────────────────────────┐
│                     YOUR PYTHON CODE                         │
│                                                              │
│  chain.invoke("Hello")                                       │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  LangSmith SDK / Callback Handler                    │     │
│  │                                                     │     │
│  │  1. on_chain_start() → creates TRACE + ROOT SPAN    │     │
│  │  2. on_llm_start()   → creates CHILD SPAN (LLM)    │     │
│  │  3. on_llm_end()     → records output, tokens, cost │     │
│  │  4. on_chain_end()   → closes root span             │     │
│  │                                                     │     │
│  │  Captures at each step:                              │     │
│  │  • Input / Output (full text)                        │     │
│  │  • Start time / End time → latency                   │     │
│  │  • Token usage (input + output + total)              │     │
│  │  • Model name, temperature, params                   │     │
│  │  • Error / Exception (if any)                        │     │
│  │  • Custom metadata & tags                            │     │
│  └───────────────────┬─────────────────────────────────┘     │
│                      │                                       │
└──────────────────────┼───────────────────────────────────────┘
                       │ HTTPS POST (async, non-blocking)
                       ▼
            ┌─────────────────────┐
            │  LangSmith Backend  │
            │  api.smith.lang...  │
            │                     │
            │  Stores in DB:      │
            │  • Trace tree       │
            │  • Token counts     │
            │  • Costs            │
            │  • Latencies        │
            │  • Feedback scores  │
            └─────────────────────┘
```

#### Trace Structure — Nesting

```
TRACE (top-level, one per user request)
│
├── SPAN: AgentExecutor (type: chain)
│   ├── input: "What's the weather in Delhi?"
│   ├── SPAN: ChatOpenAI (type: llm)
│   │   ├── input_tokens: 85
│   │   ├── output_tokens: 30
│   │   ├── output: "I'll check the weather tool"
│   │   └── latency: 1.2s
│   │
│   ├── SPAN: WeatherTool (type: tool)
│   │   ├── input: {"city": "Delhi"}
│   │   ├── output: {"temp": "32°C", "condition": "Sunny"}
│   │   └── latency: 0.5s
│   │
│   ├── SPAN: ChatOpenAI (type: llm)  ← second LLM call
│   │   ├── input_tokens: 150
│   │   ├── output_tokens: 45
│   │   ├── output: "The weather in Delhi is 32°C and sunny."
│   │   └── latency: 0.9s
│   │
│   ├── output: "The weather in Delhi is 32°C and sunny."
│   ├── total_tokens: 310
│   ├── total_cost: $0.004
│   └── total_latency: 2.6s
```

---

### 4.6 Adding Feedback (Human & Automated)

#### What is Feedback?

> Feedback allows you to **score LLM outputs** — either manually (human review) or automatically (LLM-as-judge). This is essential for improving your AI app over time.

#### Method 1 — Programmatic Feedback via SDK

```python
from langsmith import Client

client = Client()  # uses LANGCHAIN_API_KEY from env

# After a run completes, add feedback using the run_id
client.create_feedback(
    run_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # from the trace
    key="user-rating",           # feedback category name
    score=1,                     # numeric score (0-1 or custom)
    value="positive",            # optional label
    comment="Great response!"    # optional text comment
)
```

#### Method 2 — Feedback in a Pipeline (Capture run_id automatically)

```python
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

client = Client()

@traceable(name="chatbot")
def chatbot(question: str) -> str:
    # ... your LLM logic ...
    response = "This is the answer"
    
    # Get the current run's ID
    run_tree = get_current_run_tree()
    run_id = run_tree.id
    
    # Simulate user feedback (in real app, collect from UI)
    client.create_feedback(
        run_id=run_id,
        key="thumbs",
        score=1,          # 1 = thumbs up, 0 = thumbs down
    )
    
    return response
```

#### Method 3 — Automated Feedback (LLM-as-Judge)

```python
from langsmith import Client
from langchain_openai import ChatOpenAI

client = Client()
judge_llm = ChatOpenAI(model="gpt-4o-mini")

def auto_evaluate(run_id: str, question: str, answer: str):
    """Use an LLM to judge if the answer is good."""
    judgment = judge_llm.invoke(
        f"Rate this answer 1-5.\nQuestion: {question}\nAnswer: {answer}\nScore:"
    )
    score = int(judgment.content.strip()) / 5  # normalize to 0-1
    
    client.create_feedback(
        run_id=run_id,
        key="auto-quality",
        score=score,
        comment=f"LLM judge score: {judgment.content}"
    )
```

---

### 4.7 Logging Errors, Cost & Tokens

#### Errors Are Auto-Captured

```python
@traceable(name="risky_call", run_type="chain")
def risky_call():
    llm = ChatOpenAI(model="gpt-4o-mini")
    try:
        result = llm.invoke("Generate something")
        return result
    except Exception as e:
        # LangSmith automatically captures the exception
        # It will show as a RED ❌ span in the trace tree
        raise e
```

> **Errors, stack traces, and exception types** are all automatically logged in LangSmith. No extra code needed.

#### Token & Cost Tracking

LangSmith **automatically tracks** for supported models:

| Metric | Auto-Tracked? |
|--------|:------------:|
| Input Tokens | ✅ |
| Output Tokens | ✅ |
| Total Tokens | ✅ |
| Cost (USD) | ✅ (for OpenAI, Anthropic, etc.) |
| Latency | ✅ |
| Model Name | ✅ |
| Error / Exception | ✅ |

#### Manual Token Logging (for custom models)

```python
from langsmith import traceable

@traceable(name="custom_model_call", run_type="llm")
def custom_model_call(prompt: str):
    # Call your custom model
    response = my_custom_model(prompt)
    
    # Return extra metadata for LangSmith
    return {
        "output": response.text,
        "token_usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
    }
```

---

### 4.8 Prompt Hub — Pulling & Managing Prompts

#### What is the Prompt Hub?

> **LangSmith Prompt Hub** = A **centralized place to store, version, and share prompt templates**. Instead of hardcoding prompts in your code, you store them in LangSmith and **pull** them at runtime.

```
┌──────────────────────────────────────────────────┐
│              LANGSMITH PROMPT HUB                │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Prompt: "rag-answer-generator"          │    │
│  │  Version: 3 (latest)                     │    │
│  │  Template:                               │    │
│  │    System: You are a helpful assistant.   │    │
│  │    Human: Context: {context}             │    │
│  │           Question: {question}           │    │
│  │           Answer concisely.              │    │
│  │  Tags: [production, rag, v3]             │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Prompt: "email-summarizer"              │    │
│  │  Version: 1                              │    │
│  │  Template:                               │    │
│  │    Summarize this email: {email_body}    │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

#### Why Use Prompt Hub?

| Without Prompt Hub | With Prompt Hub |
|---|---|
| Prompts hardcoded in source code | Prompts stored centrally, pulled at runtime |
| Changing a prompt = code deploy | Changing a prompt = edit in UI, instant |
| No version history | Full version history with diff |
| No collaboration | Team can edit & review prompts |
| No A/B testing | Easy to swap prompt versions |

#### How to Push a Prompt to Hub

```python
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

client = Client()

# Create a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specialized in {domain}."),
    ("human", "{question}")
])

# Push to LangSmith Hub
client.push_prompt(
    "my-assistant-prompt",  # prompt name (unique identifier)
    object=prompt,
    description="General assistant prompt with domain specialization",
    is_public=False         # keep private to your org
)
```

#### How to Pull a Prompt in Code

```python
from langsmith import Client
from langchain_openai import ChatOpenAI

client = Client()

# Pull the latest version of the prompt
prompt = client.pull_prompt("my-assistant-prompt")

# Use it in a chain
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm

response = chain.invoke({
    "domain": "machine learning",
    "question": "What is backpropagation?"
})
print(response.content)
```

#### Pulling a Specific Version

```python
# Pull a specific commit/version
prompt = client.pull_prompt("my-assistant-prompt:version_hash_here")
```

---

### 4.9 Playground

#### What is the Playground?

> The **LangSmith Playground** is a **web-based UI** where you can **test prompts interactively** against different models, temperatures, and inputs — without writing any code.

#### Features of the Playground

```
┌──────────────────────────────────────────────────────────┐
│                LANGSMITH PLAYGROUND                       │
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐  │
│  │  PROMPT EDITOR      │  │  MODEL SETTINGS           │  │
│  │                     │  │                           │  │
│  │  System: You are    │  │  Model: gpt-4o-mini  ▼   │  │
│  │  a helpful...       │  │  Temperature: 0.7    ─●─ │  │
│  │                     │  │  Max Tokens: 500         │  │
│  │  Human: {question}  │  │  Top P: 1.0              │  │
│  └─────────────────────┘  └───────────────────────────┘  │
│                                                          │
│  ┌─────────────────────┐  ┌───────────────────────────┐  │
│  │  INPUT VARIABLES    │  │  OUTPUT                   │  │
│  │                     │  │                           │  │
│  │  question:          │  │  "Backpropagation is an   │  │
│  │  "What is back..."  │  │   algorithm that..."      │  │
│  └─────────────────────┘  │                           │  │
│                           │  Tokens: 120              │  │
│  [▶ Run]  [Compare]      │  Cost: $0.001             │  │
│                           │  Latency: 1.2s            │  │
│                           └───────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

#### What You Can Do:
- **Test prompts** with different inputs without code
- **Compare models** side-by-side (GPT-4 vs Claude vs Gemini)
- **Adjust parameters** (temperature, max tokens, etc.)
- **Save as a prompt** directly to the Hub
- **View token counts & costs** per run
- **Open any traced run** in the playground to replay it

---

### 4.10 Dashboard — Cost, Usage & Monitoring

The LangSmith Dashboard provides:

```
┌──────────────────────────────────────────────────────────┐
│              LANGSMITH DASHBOARD                         │
│                                                          │
│  📊 OVERVIEW (Last 7 days)                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Total    │  │ Total    │  │ Avg      │  │ Error   │ │
│  │ Traces   │  │ Cost     │  │ Latency  │  │ Rate    │ │
│  │ 12,450   │  │ $23.50   │  │ 2.1s     │  │ 0.3%    │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                          │
│  📈 COST OVER TIME                                       │
│  $8 ┤                          ╭─╮                       │
│  $6 ┤              ╭───╮    ╭──╯ │                       │
│  $4 ┤     ╭────╮╭──╯   ╰───╯    ╰──╮                    │
│  $2 ┤╭────╯    ╰╯                   ╰──                  │
│  $0 ┤────────────────────────────────────                │
│      Mon  Tue  Wed  Thu  Fri  Sat  Sun                   │
│                                                          │
│  📋 MODEL BREAKDOWN                                      │
│  ┌─────────────────┬────────┬─────────┬───────┐          │
│  │ Model           │ Calls  │ Tokens  │ Cost  │          │
│  ├─────────────────┼────────┼─────────┼───────┤          │
│  │ gpt-4o          │ 2,100  │ 1.2M    │ $18.50│          │
│  │ gpt-4o-mini     │ 8,300  │ 3.5M    │ $3.20 │          │
│  │ claude-3-sonnet │ 2,050  │ 800K    │ $1.80 │          │
│  └─────────────────┴────────┴─────────┴───────┘          │
│                                                          │
│  🔴 RECENT ERRORS                                        │
│  • RateLimitError - gpt-4o - 3 occurrences               │
│  • TimeoutError - claude-3 - 1 occurrence                 │
└──────────────────────────────────────────────────────────┘
```

### LangSmith Summary

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Zero-code tracing with LangChain | **Not open source** |
| Excellent trace tree UI | Data on LangChain's servers |
| Built-in prompt hub | Free tier limited (5K traces) |
| Token + cost auto-tracking | No self-hosting (except Enterprise) |
| Playground for testing | Vendor lock-in risk |
| Feedback & evaluation system | Expensive at scale |

---

## 5. Langfuse — Open Source Alternative

### 5.1 What is Langfuse?

> **Langfuse** is an **open-source LLM engineering platform** for tracing, prompt management, evaluation, and monitoring. It's the **#1 open-source alternative to LangSmith**.

```
┌──────────────────────────────────────────────────────────┐
│                     LANGFUSE                             │
│            "Open Source LLM Engineering Platform"         │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │  TRACING   │  │  PROMPT    │  │  EVALUATION &      │ │
│  │            │  │  MGMT      │  │  SCORING           │ │
│  │ • Traces   │  │ • Version  │  │ • Manual scores    │ │
│  │ • Spans    │  │   control  │  │ • LLM-as-judge     │ │
│  │ • Events   │  │ • Pull in  │  │ • Datasets         │ │
│  │ • I/O      │  │   code     │  │ • Benchmarks       │ │
│  │ • Scores   │  │ • Labels   │  │                    │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ ANALYTICS  │  │  RBAC &    │  │  INTEGRATIONS      │ │
│  │            │  │  SECURITY  │  │                    │ │
│  │ • Cost     │  │ • Roles    │  │ • LangChain        │ │
│  │ • Tokens   │  │ • Teams    │  │ • LlamaIndex       │ │
│  │ • Latency  │  │ • Projects │  │ • OpenAI SDK       │ │
│  │ • Users    │  │ • API Keys │  │ • Any framework    │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                          │
│  🟢 Open Source (MIT)  |  🐳 Self-Hostable  |  ☁️ Cloud  │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Langfuse vs LangSmith — Quick Comparison

| Feature | LangSmith | Langfuse |
|---------|-----------|---------|
| **Open Source** | ❌ No | ✅ Yes (MIT license) |
| **Self-Hosting** | Enterprise only | ✅ Free (Docker) |
| **Cloud Option** | ✅ | ✅ (cloud.langfuse.com) |
| **Tracing** | ✅ Excellent | ✅ Excellent |
| **Prompt Management** | ✅ Hub | ✅ Built-in |
| **Cost Tracking** | ✅ Auto | ✅ Auto |
| **Feedback/Scoring** | ✅ | ✅ |
| **RBAC** | Enterprise only | ✅ Free |
| **Evaluations** | ✅ Advanced | ✅ Good |
| **Playground** | ✅ | ✅ |
| **Data Privacy** | Data on their servers | ✅ Full control (self-host) |
| **Pricing** | Free: 5K traces/mo | Free: Unlimited (self-host) |
| **Framework Lock-in** | Tightly coupled with LangChain | Framework agnostic |

---

### 5.3 Setup — Self-Hosted (Docker Compose)

#### Option A — Docker Compose (Recommended for Local/Dev)

Create a `docker-compose.langfuse.yml`:

```yaml
version: '3.8'

services:
  langfuse:
    image: langfuse/langfuse:2
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/langfuse
      - NEXTAUTH_SECRET=my-super-secret-key-change-me   # generate a random string
      - SALT=my-salt-value-change-me                     # generate a random string
      - NEXTAUTH_URL=http://localhost:3000
      - TELEMETRY_ENABLED=true
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=langfuse
    ports:
      - "5433:5432"
    volumes:
      - langfuse_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  langfuse_pgdata:
```

```bash
# Start Langfuse
docker compose -f docker-compose.langfuse.yml up -d

# Open browser → http://localhost:3000
# Sign up → Create Org → Create Project → Get API Keys
```

#### Option B — Langfuse Cloud (No Setup)

1. Go to **https://cloud.langfuse.com**
2. Sign up → Create Project
3. Go to **Settings → API Keys**
4. Copy **Public Key** and **Secret Key**

---

### 5.4 Installation & Environment Variables

```bash
pip install langfuse langchain langchain-openai
```

```env
# .env file
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxx
LANGFUSE_HOST=http://localhost:3000          # or https://cloud.langfuse.com

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

### 5.5 Code Examples — Tracing with Langfuse

#### Method 1 — Using Langfuse's Own SDK (Low Level)

```python
from langfuse import Langfuse
import openai
from dotenv import load_dotenv

load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse()

# Create a trace
trace = langfuse.trace(
    name="simple-chat",
    user_id="user-123",           # track per user
    metadata={"environment": "dev"},
    tags=["chatbot", "v1"]
)

# Create a span for the LLM call
generation = trace.generation(
    name="gpt-4o-mini-call",
    model="gpt-4o-mini",
    input=[{"role": "user", "content": "What is AI?"}],
    model_parameters={"temperature": 0.7, "max_tokens": 500}
)

# Make the actual LLM call
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is AI?"}],
    temperature=0.7,
    max_tokens=500
)

answer = response.choices[0].message.content

# End the generation span with output & usage
generation.end(
    output=answer,
    usage={
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
        "unit": "TOKENS"
    }
)

# Add a score/feedback
trace.score(
    name="user-feedback",
    value=1,                      # 0-1 scale
    comment="Accurate answer"
)

# Flush to ensure all data is sent
langfuse.flush()

print(answer)
```

#### Method 2 — Using `@observe` Decorator (Recommended)

```python
from langfuse.decorators import observe, langfuse_context
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

@observe(as_type="generation")
def call_llm(messages: list, model: str = "gpt-4o-mini"):
    # Update the current observation with model info
    langfuse_context.update_current_observation(
        model=model,
        input=messages
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    result = response.choices[0].message.content
    
    # Update with output and usage
    langfuse_context.update_current_observation(
        output=result,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
    )
    
    return result

@observe()  # This becomes the root trace
def my_chatbot(user_message: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}
    ]
    
    # This call is automatically nested as a child span
    response = call_llm(messages)
    return response

# Run it — traces appear in Langfuse UI!
answer = my_chatbot("Explain neural networks in simple terms")
print(answer)
```

#### Method 3 — LangChain Integration (CallbackHandler)

```python
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Create Langfuse callback handler
langfuse_handler = CallbackHandler(
    public_key="pk-lf-xxxx",        # or use env vars
    secret_key="sk-lf-xxxx",
    host="http://localhost:3000",
    user_id="user-456",             # optional: track user
    session_id="session-789",       # optional: group conversations
    tags=["production", "v2"]       # optional: tags for filtering
)

# Use LangChain as normal — just pass the handler
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
chain = prompt | llm

# Pass the Langfuse handler — all spans are automatically traced!
response = chain.invoke(
    {"question": "What is LangChain?"},
    config={"callbacks": [langfuse_handler]}
)

print(response.content)

# Flush to make sure traces are sent
langfuse_handler.flush()
```

#### Method 4 — Tracing a Full RAG Pipeline

```python
from langfuse.decorators import observe, langfuse_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
from dotenv import load_dotenv

load_dotenv()

@observe(as_type="generation")
def retrieve_documents(query: str, vectorstore):
    """Retriever step — traced as its own span"""
    langfuse_context.update_current_observation(
        input=query,
        metadata={"k": 3}
    )
    docs = vectorstore.similarity_search(query, k=3)
    langfuse_context.update_current_observation(
        output=[d.page_content for d in docs]
    )
    return docs

@observe(as_type="generation")
def generate_answer(question: str, context: str):
    """LLM generation step — traced as its own span"""
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": f"Answer based on this context:\n{context}"},
        {"role": "user", "content": question}
    ]
    
    langfuse_context.update_current_observation(
        model="gpt-4o-mini",
        input=messages
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    
    langfuse_context.update_current_observation(
        output=answer,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
    )
    return answer

@observe()  # Root trace
def rag_pipeline(question: str, vectorstore):
    # Step 1: Retrieve
    docs = retrieve_documents(question, vectorstore)
    context = "\n".join([d.page_content for d in docs])
    
    # Step 2: Generate
    answer = generate_answer(question, context)
    
    # Step 3: Add score
    langfuse_context.score_current_trace(
        name="completeness",
        value=0.9,
        comment="Auto-evaluated"
    )
    
    return answer
```

---

### 5.6 Langfuse Prompt Management

```python
from langfuse import Langfuse

langfuse = Langfuse()

# ──────────────────────────────────────
# CREATE / UPDATE a prompt in Langfuse
# ──────────────────────────────────────
langfuse.create_prompt(
    name="rag-qa-prompt",
    prompt="You are an AI assistant.\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:",
    config={
        "model": "gpt-4o-mini",
        "temperature": 0.3
    },
    labels=["production"],    # mark as production-ready
    is_active=True
)

# ──────────────────────────────────────
# PULL a prompt in code
# ──────────────────────────────────────
prompt = langfuse.get_prompt("rag-qa-prompt", label="production")

# Compile with variables
compiled = prompt.compile(
    context="AI is a branch of computer science.",
    question="What is AI?"
)
print(compiled)
# Output: "You are an AI assistant.\n\nContext: AI is a branch...\n\nQuestion: What is AI?\n\nAnswer:"

# Access config
print(prompt.config)  # {"model": "gpt-4o-mini", "temperature": 0.3}
```

---

### 5.7 RBAC (Role-Based Access Control)

> **RBAC** = Control who can do what in your Langfuse organization. This is a **key reason companies choose Langfuse** over LangSmith (where RBAC is enterprise-only).

```
┌──────────────────────────────────────────────────────────┐
│              LANGFUSE RBAC MODEL                         │
│                                                          │
│  ORGANIZATION (Company Level)                            │
│  ├── Owner ──── Full access, billing, delete org         │
│  ├── Admin ──── Manage members, projects, settings       │
│  └── Member ─── Use projects assigned to them            │
│                                                          │
│  PROJECT (App Level)                                     │
│  ├── Project A: "Production Chatbot"                     │
│  │   ├── Team: [Alice (Admin), Bob (Member)]             │
│  │   ├── API Keys: pk-lf-prod-xxx, sk-lf-prod-xxx       │
│  │   └── Data: Traces, prompts, scores                   │
│  │                                                       │
│  ├── Project B: "Staging RAG Pipeline"                   │
│  │   ├── Team: [Charlie (Admin), Diana (Member)]         │
│  │   ├── API Keys: pk-lf-staging-xxx                     │
│  │   └── Data: Isolated from Project A                   │
│  │                                                       │
│  └── Project C: "Experiments"                            │
│      └── Team: [Everyone]                                │
└──────────────────────────────────────────────────────────┘
```

#### Roles & Permissions

| Permission | Owner | Admin | Member | Viewer |
|-----------|:-----:|:-----:|:------:|:------:|
| View traces | ✅ | ✅ | ✅ | ✅ |
| Create traces (API) | ✅ | ✅ | ✅ | ❌ |
| Manage prompts | ✅ | ✅ | ✅ | ❌ |
| Add scores/feedback | ✅ | ✅ | ✅ | ❌ |
| Manage API keys | ✅ | ✅ | ❌ | ❌ |
| Manage members | ✅ | ✅ | ❌ | ❌ |
| Billing & settings | ✅ | ❌ | ❌ | ❌ |
| Delete organization | ✅ | ❌ | ❌ | ❌ |

---

### 5.8 How Companies Use Langfuse (Real-World Architecture)

```
┌────────────────────────────────────────────────────────────────┐
│                COMPANY AI INFRASTRUCTURE                       │
│                                                                │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│   │ Chatbot  │   │ RAG App  │   │ AI Agent │                  │
│   │ Service  │   │ Service  │   │ Service  │                  │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘                  │
│        │              │              │                          │
│        │   Langfuse SDK (traces)     │                          │
│        └──────────┬──────────────────┘                          │
│                   ▼                                             │
│   ┌─────────────────────────────────┐                          │
│   │  LANGFUSE (Self-Hosted)         │     ┌──────────────┐     │
│   │  • Running on company's K8s     │────►│ PostgreSQL   │     │
│   │  • Behind company VPN           │     │ (Company DB) │     │
│   │  • No data leaves the company   │     └──────────────┘     │
│   └───────────────┬─────────────────┘                          │
│                   │                                             │
│        ┌──────────┼──────────┐                                  │
│        ▼          ▼          ▼                                  │
│   ┌─────────┐ ┌────────┐ ┌─────────┐                          │
│   │Dev Team │ │QA Team │ │  PM /   │                          │
│   │(Admin)  │ │(Member)│ │  Exec   │                          │
│   │         │ │        │ │(Viewer) │                          │
│   │Can:     │ │Can:    │ │Can:     │                          │
│   │• Debug  │ │• View  │ │• See    │                          │
│   │• Config │ │• Score │ │  costs  │                          │
│   │• Deploy │ │• Test  │ │• Review │                          │
│   └─────────┘ └────────┘ └─────────┘                          │
└────────────────────────────────────────────────────────────────┘
```

#### Why Companies Prefer Langfuse for Production

| Reason | Details |
|--------|---------|
| **Data Sovereignty** | All trace data stays on company servers (GDPR/HIPAA) |
| **No Vendor Lock-in** | Open source — can fork, modify, migrate anytime |
| **Cost Control** | Self-hosting = only pay for infra, no per-trace fees |
| **RBAC (Free)** | Role-based access for teams — included in open source |
| **Custom Integrations** | Direct DB access, custom dashboards, analytics |
| **Audit Trail** | Full control over logs and data retention policies |

---

### 5.9 Langfuse Dashboard & Analytics

```
┌──────────────────────────────────────────────────────────┐
│              LANGFUSE DASHBOARD                          │
│                                                          │
│  📊 TRACES OVERVIEW                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Total    │  │ Total    │  │ Avg      │  │ Unique  │ │
│  │ Traces   │  │ Cost     │  │ Latency  │  │ Users   │ │
│  │ 45,230   │  │ $142.50  │  │ 1.8s     │  │ 1,250   │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                          │
│  📈 TOKEN USAGE BY MODEL                                 │
│  ┌──────────────────────────────────────────────────┐    │
│  │  gpt-4o       ████████████████████░░░  78%       │    │
│  │  gpt-4o-mini  ████████░░░░░░░░░░░░░░  15%       │    │
│  │  claude-3.5   ████░░░░░░░░░░░░░░░░░░   7%       │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  📋 TRACE LIST (Filterable)                              │
│  ┌─────────┬────────────┬────────┬──────┬───────────┐   │
│  │ Time    │ Name       │ Tokens │ Cost │ Score     │   │
│  ├─────────┼────────────┼────────┼──────┼───────────┤   │
│  │ 2m ago  │ rag-query  │ 1,250  │$0.02│ ⭐ 0.9    │   │
│  │ 5m ago  │ chat       │ 450    │$0.01│ ⭐ 0.7    │   │
│  │ 8m ago  │ agent-task │ 3,200  │$0.05│ ❌ 0.2    │   │
│  └─────────┴────────────┴────────┴──────┴───────────┘   │
│                                                          │
│  🔍 FILTERS: [Model ▼] [User ▼] [Score ▼] [Date ▼]     │
└──────────────────────────────────────────────────────────┘
```

---

### 5.10 Full Langfuse Demo — End-to-End

```python
"""
Complete Langfuse Demo
======================
Shows: tracing, feedback, cost tracking, prompt management
"""

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────
# 1. INITIALIZE LANGFUSE
# ─────────────────────────────────────
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

# ─────────────────────────────────────
# 2. CREATE A PROMPT IN LANGFUSE
# ─────────────────────────────────────
langfuse.create_prompt(
    name="demo-assistant",
    prompt="You are a {role}. Answer the user's question.\n\nQuestion: {question}\nAnswer:",
    config={"model": "gpt-4o-mini", "temperature": 0.5},
    labels=["production"]
)

# ─────────────────────────────────────
# 3. USE THE PROMPT WITH TRACING
# ─────────────────────────────────────
@observe()
def demo_pipeline(user_question: str):
    # Pull prompt from Langfuse
    prompt_obj = langfuse.get_prompt("demo-assistant")
    compiled_prompt = prompt_obj.compile(
        role="helpful AI tutor",
        question=user_question
    )
    
    # Make LLM call (traced automatically)
    llm = ChatOpenAI(
        model=prompt_obj.config["model"],
        temperature=prompt_obj.config["temperature"]
    )
    
    langfuse_context.update_current_observation(
        input=user_question,
        metadata={"prompt_version": prompt_obj.version}
    )
    
    response = llm.invoke(compiled_prompt)
    
    langfuse_context.update_current_observation(
        output=response.content
    )
    
    # Add feedback score
    langfuse_context.score_current_trace(
        name="relevance",
        value=0.95,
        comment="Highly relevant answer"
    )
    
    return response.content

# ─────────────────────────────────────
# 4. RUN THE PIPELINE
# ─────────────────────────────────────
result = demo_pipeline("What is the difference between AI and ML?")
print(result)

# Flush all events
langfuse.flush()

print("\n✅ Check your Langfuse dashboard at http://localhost:3000")
print("   You should see: trace tree, token usage, cost, and feedback score!")
```

---

## 6. LangSmith vs Langfuse — Final Comparison

```
                    CHOOSING YOUR TRACING TOOL
                    ══════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │              DECISION FLOWCHART                      │
    │                                                     │
    │  Need Self-Hosting / Data Privacy?                  │
    │         │                                           │
    │    YES ─┤── ► LANGFUSE (self-host, open source)     │
    │         │                                           │
    │    NO ──┤── Already using LangChain heavily?        │
    │         │       │                                   │
    │         │  YES ─┤── ► LANGSMITH (tight integration) │
    │         │       │                                   │
    │         │  NO ──┤── Budget constraints?             │
    │         │       │       │                           │
    │         │       │  YES ─┤── ► LANGFUSE (free!)      │
    │         │       │       │                           │
    │         │       │  NO ──┤── ► Either works!         │
    │         │       │       │    LangSmith has better   │
    │         │       │       │    UI/UX currently.       │
    └─────────┴───────┴───────┴───────────────────────────┘
```

| Criteria | Winner |
|----------|--------|
| **Open Source** | 🏆 Langfuse |
| **Self-Hosting** | 🏆 Langfuse |
| **Data Privacy** | 🏆 Langfuse |
| **RBAC (Free)** | 🏆 Langfuse |
| **Cost at Scale** | 🏆 Langfuse |
| **UI/UX Polish** | 🏆 LangSmith |
| **LangChain Integration** | 🏆 LangSmith |
| **Prompt Hub** | 🏆 LangSmith |
| **Evaluation Features** | 🏆 LangSmith |
| **Community & Docs** | Tie |
| **Enterprise Support** | 🏆 LangSmith |

---

### Quick Reference Commands

```bash
# ─── LANGSMITH ───
pip install langsmith langchain langchain-openai
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls__xxx"
# Dashboard: https://smith.langchain.com

# ─── LANGFUSE (Self-Host) ───
pip install langfuse langchain langchain-openai
docker compose -f docker-compose.langfuse.yml up -d
export LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
export LANGFUSE_SECRET_KEY="sk-lf-xxx"
export LANGFUSE_HOST="http://localhost:3000"
# Dashboard: http://localhost:3000

# ─── LANGFUSE (Cloud) ───
# Dashboard: https://cloud.langfuse.com
```

---

> **TL;DR:** Use **LangSmith** if you want the best out-of-the-box experience with LangChain and don't mind proprietary. Use **Langfuse** if you need open source, self-hosting, data privacy, or RBAC without paying enterprise prices.