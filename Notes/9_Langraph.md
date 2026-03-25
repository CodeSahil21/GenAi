# LangGraph — Complete Guide

---

## Table of Contents

1. [What is LangGraph & What Problem Does It Solve?](#1-what-is-langgraph--what-problem-does-it-solve)
2. [Framework vs Library vs Orchestrator](#2-framework-vs-library-vs-orchestrator)
3. [LangChain vs LangGraph — What Changed?](#3-langchain-vs-langgraph--what-changed)
4. [Core Concepts — All Components](#4-core-concepts--all-components)
5. [Building Blocks — Modular Code Design](#5-building-blocks--modular-code-design)
6. [Your First Graph — Step by Step](#6-your-first-graph--step-by-step)
7. [Real Code Walkthrough — graph.py Explained](#7-real-code-walkthrough--graphpy-explained)
8. [Conditional Edges — Deep Dive](#8-conditional-edges--deep-dive)
9. [State & State Updates — How Data Flows](#9-state--state-updates--how-data-flows)
10. [ToolNode — Giving Agents Real Tools](#10-toolnode--giving-agents-real-tools)
11. [project_builder_agent.py → Reimagined as LangGraph](#11-project_builder_agentpy--reimagined-as-langgraph)
12. [Parallelism in LangGraph](#12-parallelism-in-langgraph)
13. [Debugging & Adding New Steps](#13-debugging--adding-new-steps)
14. [Quick Reference](#14-quick-reference)

---

## 1. What is LangGraph & What Problem Does It Solve?

### The Problem with Simple Chains

When you build AI apps with basic LangChain chains, everything runs in a **straight line**:

```
Input ──► Prompt ──► LLM ──► Output
```

But real AI applications need:
- **Branching** — "If the user asks about code, use GPT-4.1; otherwise use GPT-4o-mini"
- **Loops** — "Keep calling tools until the agent has enough info"
- **Human-in-the-loop** — "Pause and ask a human before executing"
- **Parallel execution** — "Search two databases at the same time"
- **State management** — "Remember what happened in step 1 when you're in step 5"

LangChain chains **cannot do this** elegantly. Enter **LangGraph**.

### What is LangGraph?

> **LangGraph** is a **graph-based orchestration framework** built on top of LangChain for creating **stateful, multi-step, cyclical AI workflows** using a **directed graph** of nodes and edges.

```
┌─────────────────────────────────────────────────────────────┐
│                     LANGGRAPH                               │
│          "Build AI Workflows as Graphs"                     │
│                                                             │
│   Instead of:  A ──► B ──► C ──► D   (linear chain)        │
│                                                             │
│   You get:     A ──► B ──┬──► C ──► E                       │
│                          │         ▲                        │
│                          └──► D ───┘   (graph with loops,   │
│                                         branches, merges)   │
└─────────────────────────────────────────────────────────────┘
```

### Key Problems LangGraph Solves

| Problem | Without LangGraph | With LangGraph |
|---------|-------------------|----------------|
| **Branching logic** | Messy if-else in chains | Clean conditional edges |
| **Loops / Retries** | Manual while loops | Built-in cycles in graph |
| **State tracking** | Pass dicts manually | Typed State auto-managed |
| **Multi-agent** | Spaghetti code | Each agent = a node |
| **Debugging** | Print statements | Visual graph + step-by-step replay |
| **Human-in-the-loop** | Not supported | Built-in interrupt/resume |
| **Parallelism** | Threading manually | Fan-out nodes auto-parallel |

---

## 2. Framework vs Library vs Orchestrator

> These terms come up a lot. Here's the difference:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   LIBRARY                                                    │
│   ────────                                                   │
│   YOU call IT. You control the flow.                         │
│   Example: requests, numpy, openai SDK                       │
│                                                              │
│   import openai                                              │
│   openai.chat.completions.create(...)  ← you call when needed│
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   FRAMEWORK                                                  │
│   ─────────                                                  │
│   IT calls YOU. It controls the flow (Inversion of Control). │
│   Example: Django, FastAPI, LangChain                        │
│                                                              │
│   @app.get("/")         ← framework calls your function      │
│   def home(): ...          when a request arrives            │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ORCHESTRATOR                                               │
│   ────────────                                               │
│   Manages the FLOW between multiple components.              │
│   Decides: What runs next? In what order? In parallel?       │
│   Example: LangGraph, Apache Airflow, Kubernetes             │
│                                                              │
│   LangGraph orchestrates:                                    │
│   • Which node to run next                                   │
│   • How state flows between nodes                            │
│   • When to branch, loop, or stop                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Where LangGraph Fits

```
                  ┌──────────────────────┐
                  │    LangGraph          │  ← Orchestrator / Framework
                  │    (Graph Engine)     │
                  └──────┬───────────────┘
                         │ uses
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ LangChain  │ │  OpenAI    │ │   Your     │  ← Libraries
   │ (prompts,  │ │  SDK       │ │   custom   │
   │  models)   │ │            │ │   code     │
   └────────────┘ └────────────┘ └────────────┘
```

> **LangGraph = Orchestrator** — it doesn't replace LangChain, it **sits on top** and orchestrates the execution flow.

---

## 3. LangChain vs LangGraph — What Changed?

```
┌──────────────────────────┬──────────────────────────────────┐
│       LANGCHAIN          │          LANGGRAPH               │
│    (Chain-based)         │       (Graph-based)              │
├──────────────────────────┼──────────────────────────────────┤
│                          │                                  │
│  prompt | llm | parser   │   StateGraph with nodes + edges  │
│                          │                                  │
│  ► Linear: A → B → C    │   ► Any shape: branches, loops   │
│  ► No cycles allowed     │   ► Cycles allowed (agents!)    │
│  ► State = just the      │   ► State = typed dict, shared  │
│    input/output          │     across all nodes             │
│  ► Hard to debug         │   ► Visual graph, step replay   │
│  ► One path only         │   ► Conditional routing          │
│  ► No pause/resume       │   ► Human-in-the-loop built in  │
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

### Analogy

| LangChain | LangGraph |
|-----------|-----------|
| Assembly line (one direction) | City road network (any direction) |
| Recipe (step 1, 2, 3...) | Flow chart (if this, go there) |
| Pipeline | State machine |

---

## 4. Core Concepts — All Components

### The 6 Building Blocks of LangGraph

```
┌─────────────────────────────────────────────────────────┐
│              LANGGRAPH — CORE COMPONENTS                │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌───────────────────────┐  │
│  │  STATE   │  │  NODE   │  │  EDGE                 │  │
│  │         │  │         │  │                       │  │
│  │ TypedDict│  │ Python  │  │ Connects nodes       │  │
│  │ shared   │  │ function│  │ • Normal edge        │  │
│  │ across   │  │ that    │  │ • Conditional edge   │  │
│  │ all nodes│  │ reads & │  │                       │  │
│  │         │  │ updates │  │                       │  │
│  │         │  │ state   │  │                       │  │
│  └─────────┘  └─────────┘  └───────────────────────┘  │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌───────────────────────┐  │
│  │  START  │  │   END   │  │  CONDITIONAL EDGE     │  │
│  │         │  │         │  │                       │  │
│  │ Entry   │  │ Exit    │  │ A function that       │  │
│  │ point   │  │ point   │  │ returns the NEXT      │  │
│  │ of the  │  │ of the  │  │ node name based on    │  │
│  │ graph   │  │ graph   │  │ current state         │  │
│  └─────────┘  └─────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Detailed Breakdown

#### 1. State (`TypedDict`)

The **shared data container** that flows through every node. Every node can read from it and write to it.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    user_message: str        # input from the user
    ai_message: str          # final AI response
    is_coding_question: bool # flag set by one node, read by another
```

> Think of State as a **shared whiteboard** — every node reads it, does work, writes results back.

#### 2. Node (Python Function)

A node is just a **regular Python function** that:
- Takes `state` as input
- Does some work (LLM call, API call, computation)
- Returns updated state

```python
def my_node(state: State):
    # Read from state
    question = state["user_message"]
    
    # Do work
    answer = call_llm(question)
    
    # Write to state & return
    state["ai_message"] = answer
    return state
```

#### 3. Edge (Connection)

Connects one node to another. Two types:

```python
# Normal edge: A always goes to B
graph.add_edge("node_a", "node_b")

# Special edges
graph.add_edge(START, "first_node")   # entry point
graph.add_edge("last_node", END)      # exit point
```

#### 4. Conditional Edge (Router)

A function that **decides which node to go to next** based on the current state.

```python
def route_function(state: State) -> Literal["path_a", "path_b"]:
    if state["is_coding_question"]:
        return "path_a"
    else:
        return "path_b"

graph.add_conditional_edges("decision_node", route_function)
```

#### 5. START & END

Special constants representing the entry and exit of the graph.

```python
from langgraph.graph import START, END

graph.add_edge(START, "first_node")   # where execution begins
graph.add_edge("last_node", END)      # where execution ends
```

#### 6. StateGraph (The Graph Builder)

The main class that wires everything together.

```python
from langgraph.graph import StateGraph

graph_builder = StateGraph(State)       # pass in your state type
graph_builder.add_node("name", func)    # add nodes
graph_builder.add_edge(START, "name")   # add edges
graph = graph_builder.compile()         # compile into runnable
result = graph.invoke(initial_state)    # execute!
```

---

## 5. Building Blocks — Modular Code Design

### Why Modularity Matters

> Each node is an **independent, reusable building block**. You can add, remove, or swap nodes without touching the rest of the graph.

```
┌─────────────────────────────────────────────────────────────┐
│           BUILDING AI APPS LIKE LEGO BLOCKS                 │
│                                                             │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│   │ Classifier│  │ Retriever │  │ Summarizer│              │
│   │   Node    │  │   Node    │  │   Node    │              │
│   │           │  │           │  │           │              │
│   │ Detects   │  │ Searches  │  │ Summarizes│              │
│   │ intent    │  │ vector DB │  │ results   │              │
│   └───────────┘  └───────────┘  └───────────┘              │
│        │              │               │                     │
│        ▼              ▼               ▼                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  GRAPH: Wire them together in ANY combination       │   │
│   │                                                     │   │
│   │  Project A: Classifier ──► Retriever ──► Summarizer │   │
│   │  Project B: Classifier ──► Summarizer (skip retriever)│  │
│   │  Project C: Retriever ──► Retriever ──► Summarizer  │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Example — Reusable Nodes

```python
# ── nodes/classifier.py ──
def classify_intent(state: State):
    """Reusable: classifies user intent"""
    result = llm.invoke(f"Classify this query: {state['user_message']}")
    state["intent"] = result.content
    return state

# ── nodes/retriever.py ──
def retrieve_docs(state: State):
    """Reusable: retrieves relevant documents"""
    docs = vectorstore.similarity_search(state["user_message"])
    state["context"] = docs
    return state

# ── nodes/generator.py ──  
def generate_answer(state: State):
    """Reusable: generates final answer"""
    response = llm.invoke(f"Context: {state['context']}\n\nQ: {state['user_message']}")
    state["ai_message"] = response.content
    return state
```

```python
# ── app.py ── Wire the blocks together
graph = StateGraph(State)
graph.add_node("classify", classify_intent)   # block 1
graph.add_node("retrieve", retrieve_docs)     # block 2
graph.add_node("generate", generate_answer)   # block 3

graph.add_edge(START, "classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
```

> **Want to add a "fact-checker" step?** Just create a new node and add one edge. No other code changes needed!

---

## 6. Your First Graph — Step by Step

### Minimal Example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ── Step 1: Define State ──
class State(TypedDict):
    name: str
    greeting: str

# ── Step 2: Define Nodes ──
def greet(state: State):
    state["greeting"] = f"Hello, {state['name']}! Welcome to LangGraph."
    return state

# ── Step 3: Build Graph ──
graph_builder = StateGraph(State)
graph_builder.add_node("greeter", greet)
graph_builder.add_edge(START, "greeter")
graph_builder.add_edge("greeter", END)

# ── Step 4: Compile ──
graph = graph_builder.compile()

# ── Step 5: Run ──
result = graph.invoke({"name": "Sahil", "greeting": ""})
print(result["greeting"])
# Output: "Hello, Sahil! Welcome to LangGraph."
```

```
Graph Visualization:

    START
      │
      ▼
  ┌────────┐
  │ greeter │
  └────┬───┘
       │
       ▼
      END
```

---

## 7. Real Code Walkthrough — graph.py Explained

> This is the actual code from `lang_graph/graph.py` in the workspace, explained line by line.

### The Full Graph Flow

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│                    graph.py — FLOW DIAGRAM                   │
│                                                              │
│                         START                                │
│                           │                                  │
│                           ▼                                  │
│                  ┌────────────────┐                          │
│                  │  detect_query  │  ← Is this a coding Q?   │
│                  │                │    Uses GPT-4o-mini       │
│                  │  Sets:         │    with structured output │
│                  │  is_coding_    │                          │
│                  │  question      │                          │
│                  └───────┬────────┘                          │
│                          │                                   │
│                 ┌────────┴────────┐  ← CONDITIONAL EDGE      │
│                 │   route_edge    │    (the router function)  │
│                 └────┬───────┬────┘                          │
│                      │       │                               │
│            is_coding │       │ NOT coding                    │
│           = True     │       │ = False                       │
│                      ▼       ▼                               │
│  ┌──────────────────────┐  ┌──────────────────────────┐     │
│  │ solve_coding_question│  │ solve_simple_question    │     │
│  │                      │  │                          │     │
│  │ Uses: GPT-4.1        │  │ Uses: GPT-4o-mini       │     │
│  │ (powerful model for  │  │ (cheap model for casual  │     │
│  │  coding problems)    │  │  chat)                   │     │
│  │                      │  │                          │     │
│  │ Sets: ai_message     │  │ Sets: ai_message         │     │
│  └──────────┬───────────┘  └────────────┬─────────────┘     │
│             │                           │                    │
│             └───────────┬───────────────┘                    │
│                         ▼                                    │
│                        END                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Code Breakdown — Section by Section

#### Section 1: Imports & Structured Output Schemas

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langsmith.wrappers import wrap_openai       # wraps OpenAI for tracing
from openai import OpenAI
from pydantic import BaseModel

# ── Pydantic Models for Structured Output ──
class DetectCallResponse(BaseModel):
    is_question_ai: bool          # LLM returns True/False as JSON

class CodingAIResponse(BaseModel):
    answer: str                   # LLM returns answer as JSON

client = wrap_openai(OpenAI())    # OpenAI client wrapped for LangSmith tracing
```

> **Why Pydantic models?** We use `response_format=DetectCallResponse` to force the LLM to return **structured JSON** instead of free text. This makes routing reliable.

#### Section 2: State Definition

```python
class State(TypedDict):
    user_message: str             # the user's input question
    ai_message: str               # the final AI response
    is_coding_question: bool      # flag: is this a coding question?
```

```
┌──────────────── STATE (shared whiteboard) ───────────────┐
│                                                          │
│  user_message: "How do I reverse a list in Python?"      │
│  ai_message: "" ──► gets filled by solver node           │
│  is_coding_question: False ──► gets set by detect_query  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### Section 3: Node — `detect_query`

```python
def detect_query(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to detect if the user's query is related
    to coding question or not.
    Return the response in specified JSON boolean only.
    """

    # Structured output — forces GPT to return {"is_question_ai": true/false}
    result = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=DetectCallResponse,
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )

    # Write to state — other nodes can now read this
    state["is_coding_question"] = result.choices[0].message.parsed.is_question_ai
    return state
```

> **What happens:** This node acts as a **classifier**. It reads the user's message, asks GPT to decide if it's a coding question, and stores the boolean in state.

#### Section 4: Conditional Edge — `route_edge`

```python
def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"     # go to coding solver
    else:
        return "solve_simple_question"     # go to simple chat
```

> **Key Syntax:** The return value **must be the exact node name** registered with `add_node()`. LangGraph uses this string to route to the correct node.

> **The `Literal` type hint** tells LangGraph (and your IDE) exactly which node names are valid return values.

#### Section 5: Solver Nodes

```python
def solve_coding_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to resolve the user query based on coding 
    problem he is facing
    """

    # Uses GPT-4.1 (more powerful, better for code)
    result = client.beta.chat.completions.parse(
        model="gpt-4.1",
        response_format=CodingAIResponse,
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )
    state["ai_message"] = result.choices[0].message.parsed.answer
    return state


def solve_simple_question(state: State):
    user_message = state.get("user_message")

    SYSTEM_PROMPT = """
    You are an AI assistant. Your job is to chat with user
    """

    # Uses GPT-4o-mini (cheaper, fine for casual chat)
    result = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        response_format=CodingAIResponse,
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_message }
        ]
    )
    state["ai_message"] = result.choices[0].message.parsed.answer
    return state
```

> **Smart model routing:** Coding questions go to the **expensive powerful model** (GPT-4.1), casual chat goes to the **cheap fast model** (GPT-4o-mini). This **saves cost** while maintaining quality.

#### Section 6: Wiring the Graph

```python
graph_builder = StateGraph(State)

# ── Register nodes (order doesn't matter) ──
graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)

# ── Wire edges ──
graph_builder.add_edge(START, "detect_query")               # START → classifier

graph_builder.add_conditional_edges("detect_query", route_edge)  # classifier → router

graph_builder.add_edge("solve_coding_question", END)        # coding solver → END
graph_builder.add_edge("solve_simple_question", END)        # simple solver → END

# ── Compile into a runnable ──
graph = graph_builder.compile()
```

#### Section 7: Running the Graph

```python
def call_graph():
    state = {
        "user_message": "Hello ji!",
        "ai_message": "",
        "is_coding_question": False
    }
    
    result = graph.invoke(state)     # runs the entire graph
    print("Final Result", result)

call_graph()
```

> **`graph.invoke(state)`** — starts at `START`, runs through nodes following edges, returns **final state** when it hits `END`.

---

## 8. Conditional Edges — Deep Dive

### Syntax

```python
graph_builder.add_conditional_edges(
    source="node_name",           # which node's OUTPUT triggers the decision
    path=route_function,           # function that returns next node name
    path_map=None                  # optional: explicit mapping (see below)
)
```

### How It Works

```
                    ┌──────────────┐
                    │ source_node  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ route_func() │  ← Your Python function
                    │              │
                    │ Reads state  │
                    │ Returns a    │
                    │ node name    │
                    └──┬───┬───┬───┘
                       │   │   │
          returns      │   │   │   returns
          "node_a"     │   │   │   "node_c"
                       ▼   │   ▼
                ┌──────┐   │  ┌──────┐
                │node_a│   │  │node_c│
                └──────┘   │  └──────┘
                           │
                    returns "node_b"
                           ▼
                        ┌──────┐
                        │node_b│
                        └──────┘
```

### Example — Three-Way Router

```python
from typing import Literal

def classify_query(state: State) -> Literal["technical", "creative", "general"]:
    intent = state["intent"]
    
    if intent == "code":
        return "technical"       # goes to technical node
    elif intent == "story":
        return "creative"        # goes to creative node
    else:
        return "general"         # goes to general node

graph.add_node("technical", handle_technical)
graph.add_node("creative", handle_creative)
graph.add_node("general", handle_general)

graph.add_conditional_edges("classifier", classify_query)
```

### With `path_map` (Rename Routes)

```python
# If your function returns short keys, map them to actual node names
graph.add_conditional_edges(
    "classifier",
    classify_query,
    path_map={
        "technical": "solve_coding_question",   # "technical" → actual node name
        "creative": "write_story_node",
        "general": "simple_chat_node"
    }
)
```

### Conditional Edge to END

```python
def should_continue(state: State) -> Literal["continue", "__end__"]:
    if state["retry_count"] > 3:
        return "__end__"          # special string: goes to END
    return "continue"

graph.add_conditional_edges("checker", should_continue)
```

> **`"__end__"`** is the string representation of `END` when used inside conditional edge functions.

---

## 9. State & State Updates — How Data Flows

### How State Flows Through Nodes

```
           Initial State
           ┌─────────────────┐
           │ user_msg: "Hi"  │
           │ ai_msg: ""      │
           │ is_code: False  │
           └────────┬────────┘
                    │
                    ▼
           ┌────────────────┐
           │  Node A        │
           │  Reads: user_msg│
           │  Writes: is_code│ ← updates one field
           └────────┬────────┘
                    │
           ┌─────────────────┐
           │ user_msg: "Hi"  │
State ──►  │ ai_msg: ""      │  ← same state, updated
           │ is_code: True   │  ← changed!
           └────────┬────────┘
                    │
                    ▼
           ┌────────────────┐
           │  Node B        │
           │  Reads: is_code │
           │  Writes: ai_msg │ ← updates another field
           └────────┬────────┘
                    │
           ┌─────────────────┐
           │ user_msg: "Hi"  │
Final ──►  │ ai_msg: "Hello!"│  ← filled in
           │ is_code: True   │
           └─────────────────┘
```

### State Update Rules

| Rule | Details |
|------|---------|
| **Read anything** | Every node can read any field in state |
| **Write anything** | Every node can write to any field |
| **Return state** | Node must return the (updated) state dict |
| **No side effects needed** | Just modify and return — LangGraph handles the rest |
| **Immutable merge** | LangGraph merges your returned dict into the running state |

### Advanced: Reducer Functions (Appending to Lists)

By default, a node's return value **replaces** the state field. But sometimes you want to **append** (e.g., chat history):

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]    # ← add = append instead of replace
    user_id: str

def node_a(state: State):
    return {"messages": ["Hello from Node A"]}    # appends, doesn't replace

def node_b(state: State):
    return {"messages": ["Hello from Node B"]}    # also appends

# After both nodes run:
# state["messages"] = ["Hello from Node A", "Hello from Node B"]
```

> **`Annotated[list, add]`** — the `add` operator tells LangGraph to **concatenate** lists instead of replacing them. This is critical for **chat history** and **document collection**.

---

## 10. ToolNode — Giving Agents Real Tools

### What is ToolNode?

> **ToolNode** is a prebuilt LangGraph node that **automatically executes tools** (functions) that an LLM decides to call. Think of it as the "executor" in a ReAct agent loop.

```
┌─────────────────────────────────────────────────────────┐
│               TOOL NODE — HOW IT WORKS                  │
│                                                         │
│   LLM says:                                             │
│   "I want to call get_weather(city='Delhi')"            │
│         │                                               │
│         ▼                                               │
│   ┌──────────┐                                          │
│   │ ToolNode │ ← Receives tool_calls from LLM           │
│   │          │   Looks up the function                   │
│   │          │   Executes it with the args               │
│   │          │   Returns result to state                 │
│   └──────────┘                                          │
│         │                                               │
│         ▼                                               │
│   Result: {"temp": "32°C", "condition": "Sunny"}        │
│   → Added to state.messages as a ToolMessage             │
│   → LLM reads this on the next turn                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Full ToolNode Example

```python
from typing_extensions import TypedDict
from typing import Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ── Step 1: Define Tools ──
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In real app, call a weather API
    weather_data = {
        "Delhi": "32°C, Sunny",
        "London": "15°C, Cloudy",
        "New York": "22°C, Partly Cloudy"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# ── Step 2: Define State ──
class State(TypedDict):
    messages: Annotated[list, add_messages]  # chat history with tool messages

# ── Step 3: Create LLM with tools bound ──
tools = [get_weather, calculate]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# ── Step 4: Define the LLM node ──
def call_llm(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ── Step 5: Build Graph ──
graph_builder = StateGraph(State)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", ToolNode(tools))    # ← Prebuilt ToolNode!

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges(
    "llm",
    tools_condition       # ← Prebuilt! Routes to "tools" if LLM made tool_calls
)
graph_builder.add_edge("tools", "llm")   # After tool execution, go back to LLM

graph = graph_builder.compile()
```

```
Graph Flow:
                  START
                    │
                    ▼
               ┌─────────┐
          ┌───►│   LLM   │◄──────────────┐
          │    └────┬─────┘               │
          │         │                     │
          │    tools_condition()           │
          │    ┌────┴────┐                │
          │    │         │                │
          │  has tool  no tool            │
          │  calls     calls             │
          │    │         │                │
          │    ▼         ▼                │
          │ ┌───────┐   END               │
          │ │ Tools │                     │
          │ │ Node  │─────────────────────┘
          │ └───────┘   (results go back to LLM)
          │
          └── (loop continues until LLM stops calling tools)
```

```python
# ── Run it! ──
result = graph.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Delhi?"}]
})

print(result["messages"][-1].content)
# "The weather in Delhi is 32°C and Sunny!"
```

### `tools_condition` Explained

```python
# tools_condition is a PREBUILT conditional edge function:
# - If the LLM's response contains tool_calls → route to "tools" node
# - If the LLM's response has NO tool_calls  → route to END

# It's equivalent to writing:
def my_tools_condition(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"
```

---

## 11. project_builder_agent.py → Reimagined as LangGraph

> The original `project_builder_agent.py` uses a **manual while-loop** to orchestrate plan → build → output steps. Here's how the **same logic** looks as a clean LangGraph workflow.

### Original Architecture (Manual Loop)

```
┌─────────────────────────────────────────────────┐
│  ORIGINAL project_builder_agent.py              │
│                                                 │
│  while True:                                    │
│    response = LLM(messages)                     │
│    parsed = json.loads(response)                │
│                                                 │
│    if step == "plan":                           │
│        print plan                               │
│        continue   ──────► loop back             │
│                                                 │
│    if step == "build":                          │
│        execute tool (create_file, etc.)         │
│        continue   ──────► loop back             │
│                                                 │
│    if step == "output":                         │
│        print final message                      │
│        break      ──────► exit loop             │
│                                                 │
│  ✗ No state management                          │
│  ✗ No visual flow                               │
│  ✗ Hard to debug which step failed              │
│  ✗ Manual JSON parsing / error handling         │
└─────────────────────────────────────────────────┘
```

### Reimagined as LangGraph

```python
"""
project_builder_agent.py — Reimagined as LangGraph

Original: Manual while-loop with JSON parsing
New: Clean graph with typed state, conditional routing, tool execution
"""

from typing_extensions import TypedDict
from typing import Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pathlib import Path
import subprocess
import json

# ═══════════════════════════════════════════════
# STATE — The shared whiteboard
# ═══════════════════════════════════════════════

class State(TypedDict):
    user_request: str              # what the user wants to build
    plan: str                      # the AI's plan
    current_step: str              # "plan" | "build" | "output"
    tool_name: str                 # which tool to execute
    tool_input: dict               # tool arguments
    tool_result: str               # tool execution result
    build_log: Annotated[list, add]  # history of all build steps
    final_message: str             # final output to show user
    messages: list                 # conversation history for the LLM

# ═══════════════════════════════════════════════
# SYSTEM PROMPT (same as original)
# ═══════════════════════════════════════════════

SYSTEM_PROMPT = """
You are WebBuilderGPT — a senior frontend developer.
Your job is to build COMPLETE, WORKING web apps.

RESPONSE FORMAT (MANDATORY):
{"step": "plan|build|output", "content": "text", "function": "tool_name", "input": {"param": "value"}}

AVAILABLE TOOLS:
- create_file: {"filepath": "path", "content": "full code"}
- create_directory: {"dirpath": "path"}
- run_command: {"command": "shell cmd", "cwd": "path"}

WORKFLOW: plan → build (repeat) → output
"""

# ═══════════════════════════════════════════════
# TOOLS (same as original)
# ═══════════════════════════════════════════════

def create_file(filepath: str, content: str) -> str:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"✓ Created {filepath}"

def create_directory(dirpath: str) -> str:
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    return f"✓ Created {dirpath}"

def run_command(command: str, cwd: str = ".") -> str:
    result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr

TOOLS = {
    "create_file": create_file,
    "create_directory": create_directory,
    "run_command": run_command,
}

# ═══════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def call_llm_node(state: State):
    """Call the LLM to get the next step."""
    if not state.get("messages"):
        state["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["user_request"]}
        ]
    
    response = llm.invoke(state["messages"])
    content = response.content.strip()
    
    # Parse JSON
    start = content.find("{")
    end = content.rfind("}") + 1
    parsed = json.loads(content[start:end])
    
    state["current_step"] = parsed.get("step", "output")
    state["plan"] = parsed.get("content", "")
    state["tool_name"] = parsed.get("function", "")
    state["tool_input"] = parsed.get("input", {})
    state["messages"].append({"role": "assistant", "content": json.dumps(parsed)})
    
    return state

def execute_tool_node(state: State):
    """Execute the tool requested by the LLM."""
    tool_func = TOOLS.get(state["tool_name"])
    if tool_func:
        result = tool_func(**state["tool_input"])
    else:
        result = f"Unknown tool: {state['tool_name']}"
    
    state["tool_result"] = result
    state["build_log"] = [f"{state['tool_name']}: {result}"]
    
    # Feed result back to conversation
    state["messages"].append({
        "role": "assistant",
        "content": json.dumps({"step": "observe", "output": result})
    })
    return state

def output_node(state: State):
    """Final output — show the result to the user."""
    state["final_message"] = f"✅ {state['plan']}"
    return state

def plan_node(state: State):
    """Show the plan and continue."""
    state["build_log"] = [f"📋 PLAN: {state['plan']}"]
    return state

# ═══════════════════════════════════════════════
# ROUTER — Conditional Edge
# ═══════════════════════════════════════════════

def route_step(state: State) -> Literal["plan_node", "execute_tool", "output_node"]:
    step = state.get("current_step", "output")
    if step == "plan":
        return "plan_node"
    elif step == "build":
        return "execute_tool"
    else:
        return "output_node"

# ═══════════════════════════════════════════════
# BUILD THE GRAPH
# ═══════════════════════════════════════════════

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("call_llm", call_llm_node)
graph_builder.add_node("plan_node", plan_node)
graph_builder.add_node("execute_tool", execute_tool_node)
graph_builder.add_node("output_node", output_node)

# Wire edges
graph_builder.add_edge(START, "call_llm")
graph_builder.add_conditional_edges("call_llm", route_step)  # LLM decides next step
graph_builder.add_edge("plan_node", "call_llm")              # plan → back to LLM
graph_builder.add_edge("execute_tool", "call_llm")           # build → back to LLM
graph_builder.add_edge("output_node", END)                    # output → done

graph = graph_builder.compile()
```

### LangGraph Version — Visual Flow

```
                         START
                           │
                           ▼
                    ┌─────────────┐
           ┌──────►│  call_llm   │◄──────────┐
           │       │             │            │
           │       │ Gets next   │            │
           │       │ step from   │            │
           │       │ the LLM     │            │
           │       └──────┬──────┘            │
           │              │                   │
           │       ┌──────▼──────┐            │
           │       │ route_step  │            │
           │       └──┬────┬──┬──┘            │
           │          │    │  │               │
           │   "plan" │    │  │ "build"       │
           │          │    │  │               │
           │          ▼    │  ▼               │
           │   ┌────────┐  │  ┌────────────┐  │
           │   │ plan   │  │  │ execute    │  │
           └───┤ _node  │  │  │ _tool      ├──┘
               └────────┘  │  └────────────┘
                           │
                    "output"│
                           ▼
                    ┌────────────┐
                    │ output     │
                    │ _node      │
                    └──────┬─────┘
                           │
                           ▼
                          END
```

### Why This Is Better

| Manual Loop (Original) | LangGraph Version |
|---|---|
| Manual while True / break | Graph handles loop automatically |
| JSON parse errors crash the loop | Each node is isolated — errors are contained |
| Hard to see what happened | `build_log` tracks every step |
| Can't pause / resume | Built-in checkpoint support |
| Can't visualize flow | Graph can be rendered as image |
| Retry logic is messy | Can add retry edges cleanly |

---

## 12. Parallelism in LangGraph

### What is Parallelism in Graphs?

> When a node has **multiple outgoing edges** that are independent, LangGraph can run them **in parallel** (fan-out), then merge results (fan-in).

```
                    START
                      │
                      ▼
                ┌───────────┐
                │  splitter  │
                └──┬──┬──┬──┘
                   │  │  │         ← FAN-OUT (parallel)
          ┌────────┘  │  └────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ search_  │ │ search_  │ │ search_  │
    │ web      │ │ database │ │ cache    │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────┬───┘────────────┘
                  │                    ← FAN-IN (merge)
                  ▼
            ┌──────────┐
            │  merger   │
            └────┬─────┘
                 │
                 ▼
                END
```

### How `graph.invoke()` Handles Parallelism

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    query: str
    results: Annotated[list, add]    # ← add reducer = append from all parallel nodes

def search_web(state: State):
    # Simulating web search
    return {"results": [f"Web result for: {state['query']}"]}

def search_database(state: State):
    # Simulating database search
    return {"results": [f"DB result for: {state['query']}"]}

def search_cache(state: State):
    # Simulating cache lookup
    return {"results": [f"Cache result for: {state['query']}"]}

def merge_results(state: State):
    combined = "\n".join(state["results"])
    return {"results": [f"MERGED:\n{combined}"]}

# Build graph with fan-out
graph_builder = StateGraph(State)

graph_builder.add_node("search_web", search_web)
graph_builder.add_node("search_database", search_database)
graph_builder.add_node("search_cache", search_cache)
graph_builder.add_node("merger", merge_results)

# Fan-out: one source → multiple targets
graph_builder.add_edge(START, "search_web")
graph_builder.add_edge(START, "search_database")
graph_builder.add_edge(START, "search_cache")

# Fan-in: multiple sources → one target
graph_builder.add_edge("search_web", "merger")
graph_builder.add_edge("search_database", "merger")
graph_builder.add_edge("search_cache", "merger")

graph_builder.add_edge("merger", END)

graph = graph_builder.compile()

# All three searches run IN PARALLEL, results merge via the `add` reducer
result = graph.invoke({"query": "LangGraph tutorial", "results": []})
```

### Parallel Execution Rules

| Rule | Details |
|------|---------|
| **Auto-parallel** | If multiple nodes have no dependency between them, they run in parallel |
| **Reducer required** | Use `Annotated[list, add]` for fields written by parallel nodes |
| **Fan-out** | Multiple edges from same source → parallel execution |
| **Fan-in** | Multiple edges to same target → waits for all before running |
| **No manual threading** | LangGraph handles the parallelism internally |

### `invoke` vs `ainvoke`

```python
# Synchronous — blocks until complete
result = graph.invoke(state)

# Asynchronous — non-blocking, use with async/await
result = await graph.ainvoke(state)

# Streaming — get results as each node completes
async for event in graph.astream(state):
    print(event)    # see each node's output as it happens
```

| Method | When to Use |
|--------|------------|
| `graph.invoke()` | Scripts, simple apps, testing |
| `await graph.ainvoke()` | FastAPI endpoints, async apps |
| `graph.astream()` | Real-time UIs, streaming responses |
| `graph.astream_events()` | Detailed event-by-event streaming |

---

## 13. Debugging & Adding New Steps

### Why LangGraph Makes Debugging Easy

```
Traditional Code              LangGraph
─────────────────              ─────────
                               
def pipeline():                Each node is isolated:
  step1()                      ┌──────────┐
  step2()     ← Where did      │  Node A  │ ← Has its own input/output
  step3()       it fail?       │  ✓ OK    │
  step4()       🤷              └─────┬────┘
                                     │
                               ┌─────▼────┐
                               │  Node B  │ ← Failed HERE!
                               │  ✗ Error │    Exact input/output visible
                               └──────────┘
```

### Visualize Your Graph

```python
# Generate a Mermaid diagram (paste into mermaid.live)
print(graph.get_graph().draw_mermaid())

# Or save as PNG (requires graphviz)
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

### Step-by-Step Execution (Debugging)

```python
# Stream each node's output to see exactly what happens
for step in graph.stream({"user_message": "Hello", "ai_message": "", "is_coding_question": False}):
    print("=" * 50)
    print(f"Node: {list(step.keys())[0]}")
    print(f"Output: {step}")
    print("=" * 50)

# Output:
# ==================================================
# Node: detect_query
# Output: {'detect_query': {'user_message': 'Hello', 'is_coding_question': False, ...}}
# ==================================================
# Node: solve_simple_question
# Output: {'solve_simple_question': {'ai_message': 'Hi there!', ...}}
# ==================================================
```

### Adding a New Step — Zero Disruption

Want to add a **"log to database"** step after the solver? Just:

```python
# 1. Write the new node
def log_to_db(state: State):
    save_to_database(state["user_message"], state["ai_message"])
    return state

# 2. Register it
graph_builder.add_node("logger", log_to_db)

# 3. Rewire ONE edge
# Before: solve_coding_question → END
# After:  solve_coding_question → logger → END
graph_builder.add_edge("solve_coding_question", "logger")  # changed
graph_builder.add_edge("solve_simple_question", "logger")  # changed
graph_builder.add_edge("logger", END)                       # new
```

> **No other code changes needed.** The classifier, router, and solver nodes are completely untouched.

---

## 14. Quick Reference

### Installation

```bash
pip install langgraph langchain langchain-openai
```

### Minimal Template

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    output: str

def my_node(state: State):
    state["output"] = f"Processed: {state['input']}"
    return state

graph = StateGraph(State)
graph.add_node("processor", my_node)
graph.add_edge(START, "processor")
graph.add_edge("processor", END)
app = graph.compile()

result = app.invoke({"input": "hello", "output": ""})
print(result["output"])  # "Processed: hello"
```

### Cheat Sheet

```
┌──────────────────────────────────────────────────────────┐
│              LANGGRAPH CHEAT SHEET                       │
│                                                          │
│  ── IMPORTS ──                                           │
│  from langgraph.graph import StateGraph, START, END      │
│  from langgraph.prebuilt import ToolNode, tools_condition│
│  from typing_extensions import TypedDict                 │
│  from typing import Annotated, Literal                   │
│  from operator import add                                │
│                                                          │
│  ── STATE ──                                             │
│  class State(TypedDict):                                 │
│      field: str                    # simple field         │
│      items: Annotated[list, add]   # append-mode field    │
│                                                          │
│  ── NODES ──                                             │
│  def my_node(state: State):                              │
│      state["field"] = "value"                            │
│      return state                                        │
│                                                          │
│  ── BUILD GRAPH ──                                       │
│  g = StateGraph(State)                                   │
│  g.add_node("name", function)                            │
│  g.add_edge(START, "name")                               │
│  g.add_edge("name", END)                                 │
│  g.add_conditional_edges("name", router_func)            │
│                                                          │
│  ── COMPILE & RUN ──                                     │
│  app = g.compile()                                       │
│  result = app.invoke(initial_state)      # sync          │
│  result = await app.ainvoke(state)       # async         │
│  for step in app.stream(state): ...      # streaming     │
│                                                          │
│  ── TOOLNODE ──                                          │
│  @tool                                                   │
│  def my_tool(arg: str) -> str: ...                       │
│  llm = ChatOpenAI(...).bind_tools([my_tool])             │
│  g.add_node("tools", ToolNode([my_tool]))                │
│  g.add_conditional_edges("llm", tools_condition)         │
│  g.add_edge("tools", "llm")                             │
│                                                          │
│  ── DEBUG ──                                             │
│  print(app.get_graph().draw_mermaid())   # visualize     │
│  for step in app.stream(state): print(step)  # step-by  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Complete Concept Map

```
                        ┌──────────────┐
                        │  LangGraph   │
                        └──────┬───────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼──────┐
     │    STATE     │   │   NODES     │   │    EDGES     │
     │              │   │             │   │              │
     │ TypedDict    │   │ Python      │   │ Normal       │
     │ Shared data  │   │ functions   │   │ Conditional  │
     │ Reducers     │   │ Read/write  │   │ START / END  │
     │ (Annotated)  │   │ state       │   │              │
     └──────────────┘   └──────┬──────┘   └──────────────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
             ┌──────▼───┐ ┌───▼────┐ ┌───▼──────┐
             │ LLM Node │ │ Tool   │ │ Custom   │
             │          │ │ Node   │ │ Logic    │
             │ Calls    │ │        │ │          │
             │ ChatGPT  │ │Executes│ │ Any      │
             │ Claude   │ │ tools  │ │ Python   │
             │ etc.     │ │ auto   │ │ code     │
             └──────────┘ └────────┘ └──────────┘
```

---

> **TL;DR:** LangGraph turns your AI app into a **visual, debuggable, modular graph**. Each step is a node, connections are edges, and state flows through everything. You get branching, loops, parallelism, and tool execution — all without manual while-loops or spaghetti code.