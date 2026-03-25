# 10. LangGraph Checkpointing, Memory, Threads & Human-in-the-Loop

---

## Table of Contents

1. [Why Do We Need Checkpointing? — The Story](#1-why-do-we-need-checkpointing--the-story)
2. [What is Checkpointing?](#2-what-is-checkpointing)
3. [Without Memory vs With Memory](#3-without-memory-vs-with-memory)
4. [Threads — Uniquely Identifying Graph Invocations](#4-threads--uniquely-identifying-graph-invocations)
5. [Config — Why We Pass It & Its Advantages](#5-config--why-we-pass-it--its-advantages)
6. [stream() vs invoke() — Deep Dive](#6-stream-vs-invoke--deep-dive)
7. [Checkpoint Storage — Where State Lives](#7-checkpoint-storage--where-state-lives)
8. [Failure Recovery & Resume — Real Scenarios](#8-failure-recovery--resume--real-scenarios)
9. [Human-in-the-Loop — Interruptions](#9-human-in-the-loop--interruptions)
10. [Mem0 AI vs LangGraph Checkpointing](#10-mem0-ai-vs-langgraph-checkpointing)
11. [Course Code — Full Line-by-Line Walkthrough](#11-course-code--full-line-by-line-walkthrough)
12. [Complete Workflow Diagrams](#12-complete-workflow-diagrams)
13. [Quick Reference](#13-quick-reference)

---

## 1. Why Do We Need Checkpointing? — The Story

### The Problem (A Simple Story)

> Imagine you're at a **restaurant** ordering food:

```
WITHOUT MEMORY (No Checkpointing)
─────────────────────────────────

You:    "I want a pizza"
Waiter: "Sure! What toppings?"

You:    "Mushrooms and olives"
Waiter: "Sorry, who are you? What pizza?" 😰

You:    "I JUST told you! Pizza with mushrooms!"
Waiter: "What's a pizza? I've never seen you before." 🤦
```

> Every time you talk to the waiter, he **forgets everything**. He has no memory. This is what happens when you run a LangGraph **without checkpointing**.

```
WITH MEMORY (Checkpointing Enabled)
────────────────────────────────────

You:    "I want a pizza"
Waiter: "Sure! What toppings?"         ← remembers you want pizza
                                         (saved: order = pizza)

You:    "Mushrooms and olives"
Waiter: "Got it! Pizza with mushrooms   ← remembers everything
         and olives. Anything else?"      (saved: order = pizza,
                                           toppings = [mushroom, olive])

You:    "Add a coke"
Waiter: "Pizza with mushrooms, olives,  ← full context preserved
         and a coke. Placing your order!"
```

> The waiter **writes down your order** after every interaction. That notebook = **Checkpoint**. The order number = **Thread ID**.

### In AI Terms

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   WITHOUT CHECKPOINTING:                                    │
│                                                             │
│   User: "My name is Sahil"                                  │
│   AI:   "Nice to meet you, Sahil!"                          │
│                                                             │
│   User: "What's my name?"                                   │
│   AI:   "I don't know your name." ← FORGOT! Each invoke()  │
│                                      starts from SCRATCH    │
│                                                             │
│   ─────────────────────────────────────────────────────     │
│                                                             │
│   WITH CHECKPOINTING:                                       │
│                                                             │
│   User: "My name is Sahil"                                  │
│   AI:   "Nice to meet you, Sahil!"   → saved to DB          │
│                                                             │
│   User: "What's my name?"                                   │
│   AI:   "Your name is Sahil!" ← REMEMBERED! State was       │
│                                  loaded from last checkpoint │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. What is Checkpointing?

> **Checkpointing** = Automatically saving a **snapshot of the graph's state** (all data in the State dict) after every node execution, so it can be **resumed, replayed, or continued** later.

### How It Works

```
┌──────────────────────────────────────────────────────────────┐
│           CHECKPOINTING — WHAT HAPPENS UNDER THE HOOD        │
│                                                              │
│   graph.invoke({"messages": [...]}, config)                  │
│                                                              │
│   Step 1: Load last checkpoint for this thread_id            │
│           ┌─────────────────────────────┐                    │
│           │  DB: thread_id = "7"        │                    │
│           │  Last state:                │                    │
│           │    messages: [msg1, msg2]   │                    │
│           └──────────────┬──────────────┘                    │
│                          │ merge with new input              │
│                          ▼                                   │
│   Step 2: Run Node "chatbot"                                 │
│           ┌─────────────┐                                    │
│           │   chatbot   │ → produces msg3                    │
│           └──────┬──────┘                                    │
│                  │                                           │
│           ✅ CHECKPOINT SAVED ─► DB now has [msg1,msg2,msg3] │
│                  │                                           │
│   Step 3: Run Node "tools" (if needed)                       │
│           ┌─────────────┐                                    │
│           │   tools     │ → produces msg4                    │
│           └──────┬──────┘                                    │
│                  │                                           │
│           ✅ CHECKPOINT SAVED ─► DB now has [msg1..msg4]     │
│                  │                                           │
│   Step 4: Run Node "chatbot" again                           │
│           ┌─────────────┐                                    │
│           │   chatbot   │ → produces msg5 (final answer)     │
│           └──────┬──────┘                                    │
│                  │                                           │
│           ✅ CHECKPOINT SAVED ─► DB now has [msg1..msg5]     │
│                  │                                           │
│                  ▼                                           │
│              Return result                                   │
│                                                              │
│   NEXT TIME user sends a message with same thread_id:        │
│   → State is loaded from DB with ALL 5 messages              │
│   → AI has full conversation history!                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Key Insight

| Without Checkpoint | With Checkpoint |
|---|---|
| State exists only in memory | State persisted to DB after every node |
| Lost when process exits | Survives restarts, crashes, deployments |
| No conversation history | Full conversation history |
| Can't resume from failure | Can resume from exact failure point |
| No human-in-the-loop possible | Human can pause, review, resume |

---

## 3. Without Memory vs With Memory

### Code Comparison

```python
# ══════════════════════════════════════════
# WITHOUT MEMORY — graph.py line 47
# ══════════════════════════════════════════
graph = graph_builder.compile()
# Every invoke() starts fresh. No history. No context.

result1 = graph.invoke({"messages": [{"role": "user", "content": "I'm Sahil"}]})
# AI: "Nice to meet you, Sahil!"

result2 = graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]})
# AI: "I don't know your name." ← FORGOT EVERYTHING


# ══════════════════════════════════════════
# WITH MEMORY — graph.py line 50-51
# ══════════════════════════════════════════
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
# Every invoke() loads previous state from DB first!

config = {"configurable": {"thread_id": "7"}}

result1 = graph.invoke(
    {"messages": [{"role": "user", "content": "I'm Sahil"}]},
    config    # ← THIS is the key — identifies the conversation
)
# AI: "Nice to meet you, Sahil!" → SAVED to DB

result2 = graph.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config    # ← SAME thread_id → loads previous messages from DB
)
# AI: "Your name is Sahil!" ← REMEMBERED!
```

### Visual Comparison

```
WITHOUT MEMORY                        WITH MEMORY
──────────────                        ───────────

invoke #1:                            invoke #1:
  state = {messages: [user_msg]}        state = {messages: [user_msg]}
  → AI responds                        → AI responds
  → state DESTROYED ❌                  → state SAVED to DB ✅

invoke #2:                            invoke #2:
  state = {messages: [new_msg]}         1. LOAD state from DB
  → AI has NO context                      (gets previous messages)
  → starts from scratch                 2. MERGE new message in
                                        3. AI has FULL context
                                        → state SAVED to DB ✅

invoke #3:                            invoke #3:
  state = {messages: [new_msg]}         1. LOAD (now has msg1-msg4)
  → still no context                    2. MERGE new message
                                        3. AI remembers EVERYTHING
                                        → state SAVED to DB ✅
```

---

## 4. Threads — Uniquely Identifying Graph Invocations

### The Concept

> Every time you start a conversation, you create a **thread**. A thread is identified by a **thread_id** — a unique string. All messages in the same conversation share the same thread_id.

```
┌──────────────────────────────────────────────────────────────┐
│                     THREADS                                  │
│                                                              │
│   THREAD = One unique conversation / session                 │
│   thread_id = The unique identifier for that conversation    │
│                                                              │
│   ┌────────────────────────────────────┐                     │
│   │  Thread ID: "7"                    │                     │
│   │  ┌──────────────────────────┐      │                     │
│   │  │ User: "Hi, I'm Sahil"   │      │                     │
│   │  │ AI:   "Hello Sahil!"    │      │                     │
│   │  │ User: "What's my name?" │      │                     │
│   │  │ AI:   "You're Sahil"    │      │                     │
│   │  └──────────────────────────┘      │                     │
│   └────────────────────────────────────┘                     │
│                                                              │
│   ┌────────────────────────────────────┐                     │
│   │  Thread ID: "42"                   │  ← Different user / │
│   │  ┌──────────────────────────┐      │    different convo   │
│   │  │ User: "Hello"           │      │                     │
│   │  │ AI:   "Hi there!"      │      │                     │
│   │  └──────────────────────────┘      │                     │
│   └────────────────────────────────────┘                     │
│                                                              │
│   ┌────────────────────────────────────┐                     │
│   │  Thread ID: "user-123-session-5"   │  ← Can be any       │
│   │  ┌──────────────────────────┐      │    unique string     │
│   │  │ User: "Translate hello" │      │                     │
│   │  │ AI:   "Hola / Bonjour" │      │                     │
│   │  └──────────────────────────┘      │                     │
│   └────────────────────────────────────┘                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### From the Course Code

```python
# main.py — line 8
config = {"configurable": {"thread_id": "7"}}
#                                        ^^^
#                          This is the thread_id!
#                          All invoke/stream calls with this config
#                          share the same conversation history.
```

### Real-World Thread ID Patterns

```python
# Pattern 1: Per user
config = {"configurable": {"thread_id": f"user-{user_id}"}}

# Pattern 2: Per session
config = {"configurable": {"thread_id": f"user-{user_id}-session-{session_id}"}}

# Pattern 3: Per conversation topic
config = {"configurable": {"thread_id": f"user-{user_id}-{conversation_id}"}}

# Pattern 4: Random UUID
import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
```

### What Happens in the Database

```
┌──────────────────────────────────────────────────────┐
│                  MongoDB                             │
│                                                      │
│  Collection: checkpoints                             │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ { thread_id: "7",                              │  │
│  │   checkpoint_id: "abc123",                     │  │
│  │   state: {                                     │  │
│  │     messages: [                                │  │
│  │       {role: "user", content: "Hi I'm Sahil"}, │  │
│  │       {role: "ai", content: "Hello Sahil!"},   │  │
│  │       {role: "user", content: "What's my name"}│  │
│  │       {role: "ai", content: "You are Sahil"}   │  │
│  │     ]                                          │  │
│  │   },                                           │  │
│  │   parent_checkpoint_id: "abc122",              │  │
│  │   timestamp: "2026-03-04T10:30:00Z"            │  │
│  │ }                                              │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ { thread_id: "42",    ← different thread       │  │
│  │   state: { messages: [...] }                   │  │
│  │ }                                              │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 5. Config — Why We Pass It & Its Advantages

### What is Config?

```python
config = {"configurable": {"thread_id": "7"}}
```

> **Config** is a dictionary you pass to every `invoke()` or `stream()` call. It tells LangGraph **which conversation** to load/save state for.

### Why Config is Required

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  WITHOUT CONFIG:                                       │
│  graph.invoke({"messages": [...]})                     │
│  → LangGraph doesn't know WHICH conversation this is   │
│  → Can't load previous state                           │
│  → Can't save state for later                          │
│  → Every call is isolated                              │
│                                                        │
│  WITH CONFIG:                                          │
│  graph.invoke({"messages": [...]}, config)              │
│  → LangGraph knows: thread_id = "7"                    │
│  → Loads previous state for thread "7" from DB         │
│  → After execution, saves new state for thread "7"     │
│  → Next call with same config continues conversation   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Advantages of Config

| Advantage | How |
|-----------|-----|
| **Multi-user support** | Different thread_id per user → isolated conversations |
| **Session management** | Same user, different sessions → different thread_ids |
| **Resumability** | Same thread_id → pick up where you left off |
| **Human-in-the-loop** | Interrupt, wait for human input, resume with same config |
| **Debugging** | Replay any thread by loading its checkpoints |
| **Audit trail** | Every thread has a full history of all states |

---

## 6. stream() vs invoke() — Deep Dive

### The Difference

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   invoke()                        stream()                   │
│   ────────                        ────────                   │
│                                                              │
│   Runs the ENTIRE graph           Runs the graph and YIELDS  │
│   and returns the FINAL           results AFTER EACH NODE    │
│   state only.                     completes.                 │
│                                                              │
│   ┌──────┐                        ┌──────┐                   │
│   │Node A│──►┌──────┐             │Node A│──► yield state    │
│   └──────┘   │Node B│──►┌──────┐  └──────┘                   │
│              └──────┘   │Node C│  ┌──────┐                   │
│                         └──┬───┘  │Node B│──► yield state    │
│                            │      └──────┘                   │
│                            ▼      ┌──────┐                   │
│                     return FINAL  │Node C│──► yield state    │
│                     state only    └──────┘                   │
│                                                              │
│   You see: just the end result    You see: every step live   │
│                                                              │
│   Use when: You only care about   Use when: You want to show │
│   the final answer                progress, debug, or stream │
│                                   to a UI in real-time       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Code Comparison

```python
# ════════════════════════════
# invoke() — Returns final state
# ════════════════════════════
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config
)
print(result["messages"][-1].content)
# Prints ONLY the final AI message


# ════════════════════════════
# stream() — Yields after EACH node
# ════════════════════════════
for event in graph.stream(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config,
    stream_mode="values"    # ← important!
):
    if "messages" in event:
        event["messages"][-1].pretty_print()
# Prints EVERY intermediate message:
# 1. User's message (after input node)
# 2. AI's tool call (after chatbot node)
# 3. Tool result (after tools node)
# 4. AI's final answer (after chatbot node again)
```

### stream_mode Options

| Mode | What It Yields | Use Case |
|------|----------------|----------|
| `"values"` | Full state after each node | Seeing all messages at each step |
| `"updates"` | Only the changes each node made | Debugging — what did this node change? |
| `"debug"` | Detailed debug info per step | Deep debugging |

### The `for event in` Pattern — Explained

```python
# From main.py — line 14-16
for event in graph_with_mongo.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values"
):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

**What happens step by step:**

```
Iteration 1:  event = state AFTER input is processed
              event["messages"] = [HumanMessage("Hello")]
              → prints: "================================ Human Message ================"
              →         "Hello"

Iteration 2:  event = state AFTER chatbot node runs
              event["messages"] = [HumanMessage, AIMessage("Hi! How can I help?")]
              → prints: "================================== Ai Message ================="
              →         "Hi! How can I help?"

(If tool calls happen, more iterations occur)

Iteration 3:  event = state AFTER tool node runs
              event["messages"] = [Human, AI(tool_call), ToolMessage("result")]
              → prints: "================================ Tool Message ================="
              →         "result"

Iteration 4:  event = state AFTER chatbot processes tool result
              event["messages"] = [Human, AI, Tool, AIMessage("Final answer")]
              → prints: "================================== Ai Message ================="
              →         "Final answer based on tool result"
```

> **Why `[-1]`?** → `event["messages"][-1]` gets only the **last (newest) message** to avoid reprinting the whole history each time.

---

## 7. Checkpoint Storage — Where State Lives

### Available Checkpointers

```
┌──────────────────────────────────────────────────────────────┐
│              CHECKPOINT STORAGE OPTIONS                       │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  MemorySaver     │  │  SQLiteSaver     │                  │
│  │  (In-Memory)     │  │  (File-based)    │                  │
│  │                  │  │                  │                  │
│  │  • For testing   │  │  • For local dev │                  │
│  │  • Lost on       │  │  • Persists to   │                  │
│  │    restart       │  │    .db file      │                  │
│  │  • Fastest       │  │  • Simple setup  │                  │
│  └──────────────────┘  └──────────────────┘                  │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │  PostgresSaver   │  │  MongoDBSaver    │  ← Course uses   │
│  │  (PostgreSQL)    │  │  (MongoDB)       │    this one!     │
│  │                  │  │                  │                  │
│  │  • Production    │  │  • Production    │                  │
│  │  • Scalable      │  │  • Scalable      │                  │
│  │  • ACID          │  │  • Flexible      │                  │
│  │  • pip install   │  │  • pip install   │                  │
│  │    langgraph-    │  │    langgraph-    │                  │
│  │    checkpoint-   │  │    checkpoint-   │                  │
│  │    postgres      │  │    mongodb       │                  │
│  └──────────────────┘  └──────────────────┘                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Setup Examples

```python
# ── Option 1: In-Memory (testing only) ──
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# ── Option 2: SQLite (local dev) ──
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = graph_builder.compile(checkpointer=checkpointer)


# ── Option 3: MongoDB (production — used in course) ──
from langgraph.checkpoint.mongodb import MongoDBSaver
MONGODB_URI = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)


# ── Option 4: PostgreSQL (production) ──
from langgraph.checkpoint.postgres import PostgresSaver
POSTGRES_URI = "postgresql://user:pass@localhost:5432/langgraph"
with PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)
```

### Docker-Compose for MongoDB (From Course)

```yaml
# docker-compose.yml
services:
  mongodb:
    image: mongo              # official MongoDB image
    restart: always           # auto-restart if crashes
    ports:
      - '27017:27017'         # expose MongoDB port
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin    # login username
      MONGO_INITDB_ROOT_PASSWORD: admin    # login password
    volumes:
      - mongodb_data:/data/db  # persist data across restarts

volumes:
  mongodb_data:                # named volume for persistence
```

```bash
# Start MongoDB
docker compose up -d

# Verify it's running
docker ps
# Should show mongo container on port 27017
```

### What Gets Stored in the Database

```
┌──────────────────────────────────────────────────────────┐
│  MongoDB Document (per checkpoint)                        │
│                                                          │
│  {                                                       │
│    "_id": "checkpoint_abc123",                           │
│    "thread_id": "7",              ← which conversation   │
│    "checkpoint_id": "abc123",     ← unique snapshot ID   │
│    "parent_id": "abc122",         ← previous checkpoint  │
│    "channel_values": {            ← the actual STATE     │
│      "messages": [                                       │
│        {"type": "human", "content": "Hi I'm Sahil"},     │
│        {"type": "ai", "content": "Hello Sahil!"},        │
│        {"type": "human", "content": "Help me with code"},│
│        {"type": "ai", "content": "Sure! What lang?"}     │
│      ]                                                   │
│    },                                                    │
│    "channel_versions": {...},     ← version tracking     │
│    "metadata": {                                         │
│      "step": 4,                   ← which step number    │
│      "source": "loop",                                   │
│      "writes": {...}              ← what this step wrote │
│    },                                                    │
│    "created_at": "2026-03-04..."                         │
│  }                                                       │
└──────────────────────────────────────────────────────────┘
```

---

## 8. Failure Recovery & Resume — Real Scenarios

### Why This Matters

> In production, things fail: API rate limits, network errors, timeouts. Without checkpointing, the **entire pipeline restarts from scratch**. With checkpointing, it **resumes from the last successful step**.

### Scenario: 4-Node Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  USE CASE: AI Banking Agent                                  │
│                                                              │
│  Node 1: input_node     — Parse user request                 │
│  Node 2: rag_node       — Search knowledge base              │
│  Node 3: transaction    — Execute bank transfer (CRITICAL!)  │
│  Node 4: confirm_node   — Send confirmation to user          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Scenario A — WITHOUT Checkpointing (Failure)

```
User: "Transfer $500 to Alice"

   Node 1: input_node ── ✅ Parsed successfully
   Node 2: rag_node   ── ✅ Found Alice's account
   Node 3: transaction ── ❌ API TIMEOUT!
   
   💥 EVERYTHING IS LOST.
   
   User has to start over.
   "Transfer $500 to Alice" (again)
   
   Node 1: input_node ── runs AGAIN (wasted work)  
   Node 2: rag_node   ── runs AGAIN (wasted work)
   Node 3: transaction ── ✅ this time (or fails again...)
   Node 4: confirm     ── ✅
```

### Scenario B — WITH Checkpointing (Failure + Recovery)

```
User: "Transfer $500 to Alice"

   Node 1: input_node ── ✅ Parsed ──► CHECKPOINT SAVED ✅
   Node 2: rag_node   ── ✅ Found  ──► CHECKPOINT SAVED ✅
   Node 3: transaction ── ❌ API TIMEOUT!
   
   💡 BUT state is saved! We know:
   • User wants to transfer $500
   • To Alice's account (found by RAG)
   • We failed at Node 3
   
   ═══ RESUME (same thread_id) ═══
   
   graph.invoke(None, config)  ← resume with no new input
                                  (or Command(resume=...))
   
   → Loads checkpoint from DB
   → Skips Node 1 (already done)
   → Skips Node 2 (already done)
   → RETRIES Node 3 ── ✅ Success!
   → Node 4: confirm ── ✅
   
   User sees: "Transfer complete!" 🎉
```

### Failure Recovery Diagram

```
                    Timeline
                    ────────
  
  Run 1:            Run 2 (resume):
  ┌──────────┐     
  │ Node 1 ✅│     
  │ SAVED    │     
  ├──────────┤     
  │ Node 2 ✅│     
  │ SAVED    │     
  ├──────────┤      ┌──────────────┐
  │ Node 3 ❌│ ───► │ LOAD from DB │
  │ CRASHED  │      │ Skip 1 & 2   │
  └──────────┘      ├──────────────┤
                    │ Node 3 ✅    │ ← retry from here
                    │ SAVED        │
                    ├──────────────┤
                    │ Node 4 ✅    │
                    │ SAVED        │
                    └──────────────┘
```

---

## 9. Human-in-the-Loop — Interruptions

### The Concept

> Sometimes the AI should **pause and ask a human** before continuing. Like how Cursor asks "Allow this terminal command?" before running it.

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   HUMAN-IN-THE-LOOP = The graph PAUSES at a certain point,  │
│   waits for human input, then RESUMES with that input.       │
│                                                              │
│   Real examples:                                             │
│   • Banking: "Confirm transfer of $5000?" → wait for yes/no │
│   • Support: "Should I escalate to manager?" → wait          │
│   • Coding:  "Run this command? rm -rf /?" → wait ⚠️         │
│   • Medical: "Prescribe medication X?" → wait for doctor     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### How Interruption Works

```
┌──────────────────────────────────────────────────────────────┐
│          INTERRUPTION FLOW (from course code graph.py)        │
│                                                              │
│   User: "I need help with my billing issue"                  │
│                                                              │
│   ┌──────────────┐                                           │
│   │   chatbot     │  LLM thinks: "I should ask a human       │
│   │   (Node)      │   support agent for help with this"      │
│   │               │                                          │
│   │   Calls:      │                                          │
│   │   human_      │                                          │
│   │   assistance_ │                                          │
│   │   tool()      │                                          │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│   ┌──────────────┐                                           │
│   │  ToolNode     │  Executes human_assistance_tool          │
│   │  (tools)      │                                          │
│   │               │  Inside the tool:                        │
│   │  interrupt()  │  ← ⚡ GRAPH STOPS HERE!                  │
│   │  is called!   │     State is saved to DB.                │
│   │               │     Program exits the stream/invoke.     │
│   └───────┬───────┘                                          │
│           │                                                  │
│    ═══════╪═══════════════════════════════════════════        │
│           │   GRAPH IS NOW PAUSED / INTERRUPTED              │
│           │   State saved in MongoDB with thread_id="7"      │
│           │                                                  │
│           │   ... time passes (minutes, hours, days) ...     │
│           │                                                  │
│           │   A HUMAN REVIEWS THE QUERY                      │
│           │   (support.py runs)                              │
│    ═══════╪═══════════════════════════════════════════        │
│           │                                                  │
│           ▼                                                  │
│   ┌──────────────┐                                           │
│   │  RESUME       │  Command(resume={"data": human_answer})  │
│   │               │                                          │
│   │  interrupt()  │  ← Returns the human's answer            │
│   │  returns      │     human_response["data"] = answer      │
│   │  human_answer │                                          │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│   ┌──────────────┐                                           │
│   │   chatbot     │  Now has the human support agent's       │
│   │   (again)     │  answer. Formulates final response       │
│   │               │  to the user.                            │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│          END                                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The `interrupt()` Function — How It Works

```python
# From graph.py — lines 11-13
from langgraph.types import interrupt

@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human."""
    
    # ⚡ THIS IS THE KEY LINE:
    human_response = interrupt({ "query": query })
    # 
    # What interrupt() does:
    # 1. STOPS the graph execution immediately
    # 2. SAVES the current state to the checkpoint DB
    # 3. SAVES the interrupt payload ({"query": query}) in state
    # 4. Returns control to the caller (stream/invoke exits)
    #
    # Later, when someone calls graph.stream(Command(resume={"data": "answer"})):
    # 5. interrupt() RETURNS with the resume data
    # 6. human_response = {"data": "answer"}
    # 7. Graph continues from this exact point
    
    return human_response["data"]  # return human's answer as tool result
```

### Customer Support Use Case — Full Example

```
AI CUSTOMER SUPPORT AGENT — Interruption Flow
══════════════════════════════════════════════

User: "I was charged twice for order #12345"

Step 1: chatbot (LLM) analyzes the issue
        → Looks serious (double charge = money!)
        → Decides: "I need human support agent approval"
        → Calls: human_assistance_tool(
              query="Customer charged twice for order #12345.
                     Should I process a refund of $49.99?"
          )

Step 2: ToolNode executes human_assistance_tool
        → interrupt() is called
        → ⚡ GRAPH STOPS
        → State saved: {
            messages: [user complaint, AI decision to escalate],
            pending_interrupt: "should I refund $49.99?"
          }

═══ MEANWHILE ═══
   
   support.py runs:
   1. Loads state from MongoDB (thread_id="7")
   2. Reads the AI's query: "Should I process a refund of $49.99?"
   3. Shows it to a human support agent
   4. Agent types: "Yes, approved. Process full refund."
   5. Sends: Command(resume={"data": "Yes, approved. Process full refund."})

Step 3: Graph RESUMES
        → interrupt() returns {"data": "Yes, approved..."}
        → human_assistance_tool returns "Yes, approved..."
        → This becomes a ToolMessage in the conversation

Step 4: chatbot runs again
        → Reads tool result: "Yes, approved. Process full refund."
        → Tells user: "I've processed a full refund of $49.99 
                        for order #12345. It will appear in 3-5 
                        business days."

Step 5: END
```

### Other Ways to Interrupt

```python
# Method 1: interrupt() inside a tool (used in course)
@tool()
def my_tool(query: str):
    response = interrupt({"query": query})
    return response["data"]

# Method 2: interrupt_before (interrupt BEFORE a node runs)
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["transaction_node"]   # pause before this node
)

# Method 3: interrupt_after (interrupt AFTER a node runs)
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["risk_assessment"]     # pause after this node
)
```

### interrupt_before Use Case — Transaction Approval

```python
# Banking agent — pause before executing a transaction
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_transaction"]  # don't run this until human approves
)

# Run 1: Graph runs nodes until it reaches execute_transaction, then STOPS
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Transfer $5000 to Bob"}]},
    config
)
# Graph is now paused BEFORE execute_transaction

# Show pending action to human
state = graph.get_state(config)
print(state.next)  # ("execute_transaction",) — this is what would run next

# Human approves — resume!
result = graph.invoke(None, config)  # None = no new input, just continue
# execute_transaction runs → confirm_node runs → END
```

---

## 10. Mem0 AI vs LangGraph Checkpointing

### What is Mem0?

> **Mem0** is an external **long-term memory layer** for AI apps. It extracts and stores facts about users across sessions.

### Key Differences

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   LANGGRAPH CHECKPOINTING              MEM0 AI              │
│   ─────────────────────                ────────              │
│                                                              │
│   Stores: Full graph STATE             Stores: FACTS about   │
│   (messages, variables,                user (extracted by     │
│    tool results, flags)                an LLM from convos)   │
│                                                              │
│   Scope: One conversation              Scope: Across ALL     │
│   (per thread_id)                      conversations         │
│                                                              │
│   Purpose: Continue a                  Purpose: Remember     │
│   specific conversation                user preferences      │
│                                        long-term             │
│                                                              │
│   Format: Raw state dict               Format: Structured    │
│   (exact messages stored)              memories (extracted)  │
│                                                              │
│   Example:                             Example:              │
│   "What did I ask 5 min ago?"          "I know you prefer    │
│   → loads exact messages               Python and use VS     │
│                                        Code"                 │
│                                                              │
│   Analogy: A TAPE RECORDER             Analogy: A MEMORY     │
│   Records everything verbatim          Remembers key facts   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Feature | LangGraph Checkpointing | Mem0 AI |
|---------|------------------------|---------|
| **What's stored** | Full state (all messages verbatim) | Extracted facts/preferences |
| **Scope** | Per thread (one conversation) | Per user (across all conversations) |
| **Storage** | DB (Mongo/Postgres/SQLite) | Mem0 cloud or self-hosted |
| **retrieval** | Load entire thread state | Semantic search for relevant memories |
| **Use case** | Continue conversation, failure recovery | Personalization across sessions |
| **Works with** | LangGraph only | Any framework |
| **Example** | "Recall what I said 3 messages ago" | "You mentioned liking spicy food last week" |

### When to Use What

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Use CHECKPOINTING when you need:                        │
│  ✓ Conversation continuity (chatbot)                     │
│  ✓ Failure recovery (resume from crash)                  │
│  ✓ Human-in-the-loop (pause/resume)                     │
│  ✓ Debugging (replay a conversation)                     │
│                                                          │
│  Use MEM0 when you need:                                 │
│  ✓ Cross-session memory ("You told me last week...")     │
│  ✓ User preferences (language, style, interests)         │
│  ✓ Facts about the user (job, location, preferences)     │
│  ✓ Personalization across different conversations         │
│                                                          │
│  Use BOTH for the best experience:                       │
│  ✓ Checkpointing for conversation state                  │
│  ✓ Mem0 for long-term user knowledge                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 11. Course Code — Full Line-by-Line Walkthrough

### File 1: `graph.py` — The Graph Definition

```python
# ═══════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════

from typing import Annotated
# Annotated lets us attach metadata to types.
# Used here: Annotated[list, add_messages]
# → tells LangGraph to APPEND messages instead of replacing

from typing_extensions import TypedDict
# TypedDict = a dict with predefined key names and types
# Used for defining our State schema

from langgraph.graph.message import add_messages
# add_messages = a REDUCER function
# When a node returns {"messages": [new_msg]}, instead of
# replacing the entire list, it APPENDS the new message
# Also handles deduplication by message ID

from langchain.chat_models import init_chat_model
# Factory function to create any chat model
# init_chat_model(model_provider="openai", model="gpt-4.1")
# → returns a ChatOpenAI instance
# Advantage: can switch providers without changing code

from langgraph.graph import StateGraph, START, END
# StateGraph = the graph builder class
# START = special node marking entry point
# END = special node marking exit point

from langchain_core.tools import tool
# @tool decorator: converts a Python function into a
# LangChain Tool that an LLM can call

from langgraph.types import interrupt
# interrupt() = PAUSES the graph, saves state, exits
# Used for human-in-the-loop

from langgraph.prebuilt import ToolNode, tools_condition
# ToolNode = prebuilt node that EXECUTES tool calls from LLM
# tools_condition = prebuilt router that checks if LLM
#                   wants to call tools or not


# ═══════════════════════════════════════════════════════════════
# TOOL DEFINITION — Human Assistance
# ═══════════════════════════════════════════════════════════════

@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human."""
    #
    # This docstring is IMPORTANT — the LLM reads it to decide
    # WHEN to call this tool. It acts as the tool's description.
    #
    # When the LLM decides it needs human help, it calls this tool
    # with a query like: "Customer wants a refund of $50. Approve?"

    human_response = interrupt({ "query": query })
    # interrupt() does:
    # 1. Package {"query": query} as the interrupt payload
    # 2. STOP the graph immediately
    # 3. Save ALL state to the checkpoint DB
    # 4. Return control to the caller (stream/invoke exits)
    #
    # LATER, when support.py sends Command(resume={"data": "approved"}):
    # 5. interrupt() RETURNS {"data": "approved"}
    # 6. Graph continues from this exact line

    return human_response["data"]
    # The human's answer becomes the tool's return value
    # → added as a ToolMessage to the conversation
    # → LLM reads it in the next chatbot turn


tools = [human_assistance_tool]
# List of all tools the LLM can use
# Could add more: [human_assistance_tool, search_tool, calculator]


# ═══════════════════════════════════════════════════════════════
# LLM SETUP
# ═══════════════════════════════════════════════════════════════

llm = init_chat_model(model_provider="openai", model="gpt-4.1")
# Creates a ChatOpenAI(model="gpt-4.1") instance
# model_provider="openai" → uses OpenAI API
# Requires OPENAI_API_KEY in environment

llm_with_tools = llm.bind_tools(tools=tools)
# .bind_tools() tells the LLM about available tools
# The LLM can now generate tool_calls in its responses
# Instead of just text, it can say:
# "I want to call human_assistance_tool(query='...')"


# ═══════════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[list, add_messages]
    #
    # messages = the conversation history
    # Annotated[list, add_messages] means:
    #   - Type: list
    #   - Reducer: add_messages
    #   - When a node returns {"messages": [new_msg]},
    #     it APPENDS to the existing list (not replaces)
    #   - Also deduplicates by message ID
    #
    # Without Annotated[list, add_messages]:
    #   Node returns {"messages": [msg3]}
    #   → state["messages"] = [msg3]  ← REPLACED! Lost msg1, msg2!
    #
    # With Annotated[list, add_messages]:
    #   Node returns {"messages": [msg3]}
    #   → state["messages"] = [msg1, msg2, msg3]  ← APPENDED! ✅


# ═══════════════════════════════════════════════════════════════
# NODE — Chatbot
# ═══════════════════════════════════════════════════════════════

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Sends ENTIRE conversation history to the LLM
    # LLM returns either:
    #   a) A regular text response (AIMessage)
    #   b) A tool call (AIMessage with tool_calls)

    assert len(message.tool_calls) <= 1
    # SAFETY CHECK: Ensure the LLM calls at most 1 tool per turn
    # Why? Our graph handles one tool call at a time
    # If it called 2+ tools, things could get messy
    # assert = if condition is False, crash with AssertionError
    # In production, you'd handle this more gracefully

    return {"messages": [message]}
    # Return the new message to be APPENDED to state
    # Thanks to add_messages reducer, this doesn't replace history


# ═══════════════════════════════════════════════════════════════
# TOOL NODE (prebuilt)
# ═══════════════════════════════════════════════════════════════

tool_node = ToolNode(tools=tools)
# ToolNode does all of this automatically:
# 1. Reads the last AIMessage from state["messages"]
# 2. Checks if it has tool_calls
# 3. For each tool_call:
#    a. Looks up the function by name (e.g., "human_assistance_tool")
#    b. Calls it with the arguments from the tool_call
#    c. Creates a ToolMessage with the result
# 4. Returns {"messages": [ToolMessage(...)]}
#
# Without ToolNode, you'd write all this manually!


# ═══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

graph_builder = StateGraph(State)
# Create a new graph builder with our State schema

graph_builder.add_node("chatbot", chatbot)
# Register the chatbot function as a node named "chatbot"

graph_builder.add_node("tools", tool_node)
# Register the ToolNode as a node named "tools"

graph_builder.add_edge(START, "chatbot")
# Entry point: when graph starts, go to "chatbot" first

graph_builder.add_conditional_edges(
    "chatbot",      # after chatbot runs...
    tools_condition, # call this function to decide next step
)
# tools_condition checks the LLM's response:
#   - If response has tool_calls → return "tools" (go to tool node)
#   - If response has NO tool_calls → return "__end__" (go to END)
#
# Equivalent to writing:
# def tools_condition(state):
#     last_msg = state["messages"][-1]
#     if last_msg.tool_calls:
#         return "tools"
#     return "__end__"

graph_builder.add_edge("tools", "chatbot")
# After tools execute, go back to chatbot
# Why? The LLM needs to READ the tool result and formulate
# a response (or call another tool)
# This creates a LOOP: chatbot → tools → chatbot → tools → ...
# The loop only exits when the LLM stops making tool calls

graph_builder.add_edge("chatbot", END)
# NOTE: This edge + the conditional edge both exist on "chatbot"
# The conditional edge takes priority when it applies
# This edge is the fallback when tools_condition returns "__end__"


# ═══════════════════════════════════════════════════════════════
# COMPILATION
# ═══════════════════════════════════════════════════════════════

graph = graph_builder.compile()
# Compile WITHOUT checkpointer = no memory
# Every invoke() starts fresh, no conversation history

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
# Factory function that compiles WITH a checkpointer
# This enables:
# ✅ Conversation memory (per thread_id)
# ✅ State persistence in DB
# ✅ Failure recovery
# ✅ Human-in-the-loop (interrupt/resume)
```

### Graph Visualization

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│              graph.py — COMPLETE FLOW                    │
│                                                          │
│                       START                              │
│                         │                                │
│                         ▼                                │
│                  ┌─────────────┐                         │
│           ┌─────►│   chatbot   │◄────────────┐           │
│           │      │             │             │           │
│           │      │ LLM thinks  │             │           │
│           │      │ & responds  │             │           │
│           │      └──────┬──────┘             │           │
│           │             │                    │           │
│           │      tools_condition()           │           │
│           │      ┌──────┴──────┐             │           │
│           │      │             │             │           │
│           │   has tool      no tool          │           │
│           │   calls         calls            │           │
│           │      │             │             │           │
│           │      ▼             ▼             │           │
│           │  ┌────────┐      END             │           │
│           │  │ tools   │                     │           │
│           │  │ (Tool   │                     │           │
│           │  │  Node)  │                     │           │
│           │  │         │                     │           │
│           │  │ If tool=│                     │           │
│           │  │ human_  │                     │           │
│           │  │ assist: │                     │           │
│           │  │ ⚡INTER- │                     │           │
│           │  │  RUPT!  │                     │           │
│           │  └────┬────┘                     │           │
│           │       │                          │           │
│           │       └──────────────────────────┘           │
│           │       (tools → chatbot loop)                 │
│           │                                              │
│           │  If interrupted:                             │
│           │  ═══════════════                             │
│           │  Graph STOPS. State saved to DB.             │
│           │  Later: Command(resume=data)                 │
│           └── Graph RESUMES from interrupt point         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

### File 2: `main.py` — The Chat Interface (Normal Chat with Memory)

```python
# ═══════════════════════════════════════════════════════════════
# main.py — Chat interface with MongoDB checkpointing
# ═══════════════════════════════════════════════════════════════

from .graph import create_chat_graph
# Import the factory function from graph.py
# .graph = relative import (same package/folder)
# create_chat_graph takes a checkpointer and returns a compiled graph

from dotenv import load_dotenv
# Loads environment variables from .env file
# (e.g., OPENAI_API_KEY)

from langgraph.checkpoint.mongodb import MongoDBSaver
# MongoDB checkpointer implementation
# Saves/loads state from a MongoDB database
# Install: pip install langgraph-checkpoint-mongodb

load_dotenv()
# Actually load the .env file now
# After this, os.getenv("OPENAI_API_KEY") works

MONGODB_URI = "mongodb://admin:admin@localhost:27017"
# Connection string for MongoDB
# admin:admin = username:password (from docker-compose.yml)
# localhost:27017 = host:port (mapped in docker-compose.yml)

config = {"configurable": {"thread_id": "7"}}
# ═══════════════════════════════════════════════════════════
# THIS IS THE THREAD ID!
# "7" = unique identifier for this conversation
#
# All messages with config thread_id="7" belong to the SAME
# conversation. Change to "8" = entirely new conversation.
#
# In a real app:
#   config = {"configurable": {"thread_id": f"user-{user_id}"}}
# ═══════════════════════════════════════════════════════════

def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        # Creates a MongoDBSaver connected to our MongoDB
        # `with` statement ensures proper cleanup (connection closing)
        #
        # What MongoDBSaver does:
        # • Creates collections in MongoDB for checkpoints
        # • Saves state after every node execution
        # • Loads state when invoke/stream is called with a thread_id

        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        # Compile the graph WITH the MongoDB checkpointer
        # Now this graph has memory!

        while True:
            user_input = input("> ")
            # Read user input from terminal

            for event in graph_with_mongo.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="values"
            ):
            # ── STREAM BREAKDOWN ──
            #
            # Argument 1: {"messages": [new_user_message]}
            #   The new input. Thanks to add_messages reducer,
            #   this gets APPENDED to existing messages in state.
            #
            # Argument 2: config
            #   {"configurable": {"thread_id": "7"}}
            #   Tells LangGraph which conversation to load/save.
            #   FIRST CALL: no existing state → starts fresh
            #   LATER CALLS: loads all previous messages from MongoDB
            #
            # Argument 3: stream_mode="values"
            #   Yield the FULL state after each node completes.
            #   (vs "updates" which yields only the changes)
            #
            # The stream yields events in this order:
            #   1. After input processing → state with [user_msg]
            #   2. After chatbot node    → state with [user_msg, ai_msg]
            #   3. After tools node      → state with [..., tool_result]
            #   4. After chatbot again   → state with [..., final_ai_msg]

                if "messages" in event:
                    event["messages"][-1].pretty_print()
                    # Only print the LAST (newest) message
                    # [-1] = last element in the list
                    #
                    # .pretty_print() formats it nicely:
                    # "================================ Human Message ================"
                    # "Hello!"
                    # "================================== Ai Message ================="
                    # "Hi there! How can I help you today?"

init()
```

### What Happens When You Run main.py

```
Terminal Session (thread_id = "7"):
──────────────────────────────────

> Hi, I'm Sahil
================================ Human Message =================================
Hi, I'm Sahil
================================== Ai Message ==================================
Hello Sahil! How can I help you today?

     └──► MongoDB now stores: [HumanMsg("Hi, I'm Sahil"), AIMsg("Hello Sahil!")]

> What's my name?
================================ Human Message =================================
What's my name?
================================== Ai Message ==================================
Your name is Sahil! 😊

     └──► MongoDB now stores: [msg1, msg2, HumanMsg("What's my name?"), AIMsg("Sahil")]
           ↑ It REMEMBERS because it loaded msg1 & msg2 from DB first!

> I need help with my billing issue
================================ Human Message =================================
I need help with my billing issue
================================== Ai Message ==================================
(AI decides to call human_assistance_tool)
================================= Tool Call ====================================
human_assistance_tool(query="Customer Sahil needs help with billing issue...")

     └──► ⚡ INTERRUPTED! Graph stops. State saved. Program waits.
          Need to run support.py to resume.
```

---

### File 3: `support.py` — Human Resume Interface

```python
# ═══════════════════════════════════════════════════════════════
# support.py — Human agent interface for handling interruptions
#
# This file is run by a HUMAN SUPPORT AGENT after the chatbot
# has interrupted (paused) asking for human assistance.
# ═══════════════════════════════════════════════════════════════

from .graph import create_chat_graph
# Same graph definition — we need it to resume

from dotenv import load_dotenv
import json
# json module: for parsing the tool call arguments

from langgraph.types import Command
# Command = special class to send instructions to a paused graph
# Command(resume={"data": "..."}) = "here's the human's answer, continue"

from langgraph.checkpoint.mongodb import MongoDBSaver
# Same checkpointer — connects to the SAME MongoDB
# to load the interrupted state

load_dotenv()
MONGODB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "7"}}
# ══════════════════════════════════════════════════════════
# SAME thread_id as main.py!
# "7" = the conversation that was interrupted
# This is how we resume the SAME conversation
# ══════════════════════════════════════════════════════════

def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)

        # ── STEP 1: Load the interrupted state ──
        state = graph_with_mongo.get_state(config=config)
        # get_state() loads the LATEST checkpoint for thread "7"
        # from MongoDB. This is the state at the moment of interruption.
        #
        # state.values = {"messages": [...all messages so far...]}
        # state.next = ("tools",)  ← what node was about to run / running
        # state.tasks = [...]      ← pending tasks with interrupt info

        # ── STEP 2: Extract the AI's question to the human ──
        last_message = state.values['messages'][-1]
        # The last message is the AIMessage that contains the tool call
        # It looks like:
        # AIMessage(
        #   content="",
        #   tool_calls=[{
        #     "name": "human_assistance_tool",
        #     "args": {"query": "Customer needs help with billing..."},
        #     "id": "call_abc123"
        #   }]
        # )

        tool_calls = last_message.additional_kwargs.get("tool_calls", [])
        # additional_kwargs contains the raw tool_calls from OpenAI
        # Format: [{"id": "call_xxx", "type": "function",
        #           "function": {"name": "...", "arguments": "{...}"}}]

        user_query = None

        for call in tool_calls:
            if call.get("function", {}).get("name") == "human_assistance_tool":
                # Found the human assistance tool call!
                args = call["function"].get("arguments", "{}")
                # arguments is a JSON STRING: '{"query": "Customer needs..."}'

                try:
                    args_dict = json.loads(args)
                    # Parse JSON string → Python dict
                    user_query = args_dict.get("query")
                    # Extract the query the AI is asking about
                except json.JSONDecodeError:
                    print("Failed to decode function arguments.")

        # ── STEP 3: Show the query to the human agent ──
        print("User is Trying to Ask:", user_query)
        # Output: "User is Trying to Ask: Customer Sahil needs help
        #          with billing issue. Should I process a refund?"

        ans = input("Resolution > ")
        # Human support agent types their answer
        # e.g., "Yes, approved. Process refund of $29.99"

        # ── STEP 4: Resume the graph with human's answer ──
        resume_command = Command(resume={"data": ans})
        # Command(resume=...) tells LangGraph:
        # "The interrupt() call should return THIS value"
        #
        # In graph.py:
        #   human_response = interrupt({"query": query})
        #   ↑ This will now return {"data": "Yes, approved..."}
        #
        #   return human_response["data"]
        #   ↑ This returns "Yes, approved..." as the tool result

        for event in graph_with_mongo.stream(
            resume_command,    # ← NOT a message, it's a RESUME command
            config,            # ← same thread_id to resume the right convo
            stream_mode="values"
        ):
            if "messages" in event:
                event["messages"][-1].pretty_print()
        # The graph RESUMES:
        # 1. interrupt() returns {"data": ans}
        # 2. human_assistance_tool returns ans
        # 3. ToolNode creates ToolMessage(content=ans)
        # 4. chatbot node runs again with tool result
        # 5. LLM formulates final response to user
        # 6. Prints the final AI message

init()
```

### What Happens When You Run support.py

```
Terminal Session (support agent):
─────────────────────────────────

User is Trying to Ask: Customer Sahil needs help with billing issue.
                       Should I process a refund?

Resolution > Yes, approved. Process full refund of $29.99.

================================= Tool Message =================================
Yes, approved. Process full refund of $29.99.
================================== Ai Message ==================================
Great news, Sahil! I've received approval from our support team. 
Your refund of $29.99 has been processed and will appear in your 
account within 3-5 business days. Is there anything else I can help with?
```

---

## 12. Complete Workflow Diagrams

### Full System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              COMPLETE SYSTEM ARCHITECTURE                     │
│                                                              │
│  ┌──────────┐          ┌──────────────────────┐              │
│  │ main.py  │          │   graph.py            │              │
│  │          │          │                      │              │
│  │ User     │───invoke/│  START                │              │
│  │ Chat     │  stream  │    │                  │              │
│  │ Loop     │──────────►  chatbot ◄──┐         │              │
│  │          │          │    │        │         │              │
│  │          │          │ tools_cond  │         │              │
│  │          │          │   / \       │         │              │
│  │          │          │  ▼   ▼      │         │              │
│  │          │          │ tools END   │         │              │
│  │          │          │  │          │         │              │
│  │          │          │  └──────────┘         │              │
│  └──────────┘          └───────────┬──────────┘              │
│                                    │                         │
│                                    │ checkpoint              │
│                                    │ save/load               │
│                                    ▼                         │
│  ┌─────────────┐          ┌──────────────────┐               │
│  │ support.py  │          │    MongoDB        │               │
│  │             │          │                  │               │
│  │ Human       │          │  thread_id: "7"  │               │
│  │ Resume      │──resume──│  messages: [...]  │               │
│  │ Interface   │          │  checkpoint: ... │               │
│  └─────────────┘          └──────────────────┘               │
│                                                              │
│  ┌──────────────────────────────────────────────┐            │
│  │              docker-compose.yml               │            │
│  │                                              │            │
│  │  MongoDB container on port 27017              │            │
│  │  Credentials: admin/admin                     │            │
│  │  Volume: mongodb_data (persists data)         │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### Timeline — Normal Chat (No Interruption)

```
main.py                    graph               MongoDB
────────                   ─────               ───────

> "Hi I'm Sahil"
  │
  └──stream()───────────►  chatbot node
                           │ LLM: "Hello Sahil!"
                           │
                           │ tools_condition → no tools → END
                           │
                           └──save──────────────► {thread:"7",
                                                   msgs: [hi, hello]}
  ◄──event: AI Message────

> "What's my name?"
  │
  └──stream()───────────►  ◄──load──────────── {thread:"7",
                           │                    msgs: [hi, hello]}
                           │ (now has context!)
                           │
                           chatbot node
                           │ LLM: "Your name is Sahil!"
                           │ tools_condition → END
                           │
                           └──save──────────────► {thread:"7",
                                                   msgs: [..., "name?", "Sahil!"]}
  ◄──event: AI Message────
```

### Timeline — With Interruption

```
main.py                    graph               MongoDB           support.py
────────                   ─────               ───────           ──────────

> "Help with billing"
  │
  └─stream()────────────►  chatbot node
                           │ LLM calls human_
                           │ assistance_tool
                           │
                           tools node
                           │ interrupt()!
                           │
                           ⚡ STOPS ──save────► {thread:"7",
                                                 msgs: [...],
                                                 interrupt: pending}
  ◄──stream ENDS──────────
  (user just sees the
   tool call message,
   program keeps running
   waiting for next input)
                                                                 get_state()
                                                 ◄──load────── reads pending
                                                                 interrupt
                                                                 │
                                                                 "Resolution > "
                                                                 agent types answer
                                                                 │
                                                                 Command(resume=
                                                                  {"data": answer})
                                                                 │
                           ◄─────────resume──────────────────────┘
                           │
                           tools node continues
                           │ interrupt() returns answer
                           │ tool returns answer
                           │
                           chatbot node (again)
                           │ LLM reads tool result
                           │ LLM: "Refund processed!"
                           │ tools_condition → END
                           │
                           └──save──────────────► {thread:"7",
                                                   msgs: [...complete]}
                                                                 │
                                                 ◄──event────────┘
                                                                 prints final answer
```

---

## 13. Quick Reference

### Installation

```bash
pip install langgraph langchain langchain-openai
pip install langgraph-checkpoint-mongodb    # for MongoDB
# OR
pip install langgraph-checkpoint-postgres   # for PostgreSQL
```

### Checkpointing Cheat Sheet

```
┌──────────────────────────────────────────────────────────────┐
│            CHECKPOINTING CHEAT SHEET                         │
│                                                              │
│  ── ENABLE CHECKPOINTING ──                                  │
│  graph = graph_builder.compile(checkpointer=checkpointer)    │
│                                                              │
│  ── CREATE CHECKPOINTERS ──                                  │
│  MemorySaver()                         # in-memory (testing) │
│  SqliteSaver.from_conn_string("f.db")  # SQLite (dev)        │
│  MongoDBSaver.from_conn_string(uri)    # MongoDB (prod)      │
│  PostgresSaver.from_conn_string(uri)   # Postgres (prod)     │
│                                                              │
│  ── THREAD CONFIG ──                                         │
│  config = {"configurable": {"thread_id": "unique-id"}}       │
│                                                              │
│  ── INVOKE / STREAM ──                                       │
│  result = graph.invoke(input, config)       # final only     │
│  for event in graph.stream(input, config):  # step by step   │
│                                                              │
│  ── INTERRUPT (Human-in-the-Loop) ──                         │
│  from langgraph.types import interrupt                       │
│  response = interrupt({"query": "..."})     # pauses graph   │
│                                                              │
│  ── RESUME ──                                                │
│  from langgraph.types import Command                         │
│  graph.stream(Command(resume={"data": ans}), config)         │
│                                                              │
│  ── GET STATE ──                                             │
│  state = graph.get_state(config)    # load current state     │
│  state.values["messages"]           # conversation history   │
│  state.next                         # what node is next      │
│                                                              │
│  ── INTERRUPT BEFORE/AFTER ──                                │
│  graph_builder.compile(                                      │
│      checkpointer=cp,                                        │
│      interrupt_before=["node_name"],  # pause BEFORE node    │
│      interrupt_after=["node_name"],   # pause AFTER node     │
│  )                                                           │
│                                                              │
│  ── assert (SAFETY CHECK) ──                                 │
│  assert len(message.tool_calls) <= 1                         │
│  # Crashes if condition is False                             │
│  # Used to ensure LLM makes at most 1 tool call per turn    │
│                                                              │
│  ── add_messages REDUCER ──                                  │
│  messages: Annotated[list, add_messages]                     │
│  # APPENDS new messages instead of replacing the list        │
│  # Critical for conversation history!                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Summary Concept Map

```
                    ┌──────────────────┐
                    │  CHECKPOINTING   │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼──────┐
   │   MEMORY    │   │  THREADS    │   │ INTERRUPTS   │
   │             │   │             │   │              │
   │ State saved │   │ thread_id   │   │ interrupt()  │
   │ after each  │   │ identifies  │   │ pauses graph │
   │ node in DB  │   │ each convo  │   │              │
   │             │   │ uniquely    │   │ Command(     │
   │ Enables:    │   │             │   │  resume=...) │
   │ • History   │   │ config =    │   │ resumes it   │
   │ • Recovery  │   │ {thread_id} │   │              │
   │ • Resume    │   │             │   │ Human-in-    │
   └─────────────┘   └─────────────┘   │ the-loop     │
                                       └──────────────┘
```

---

> **TL;DR:** Checkpointing = saving graph state to a database after every node. Thread ID = unique conversation identifier. Together they give you: conversation memory, failure recovery, and human-in-the-loop. The course code uses MongoDB to store checkpoints, `interrupt()` to pause for human input, and `Command(resume=...)` to continue after a human responds.