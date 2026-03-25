# MCP — Model Context Protocol

> **Owner:** Anthropic (creators of Claude)  
> **What:** An **open standard protocol** that defines how AI applications communicate with external tools and data sources.  
> **Analogy:** MCP is to AI tools what **USB is to peripherals** — a universal plug-and-play standard.

---

## Table of Contents

1. [What Is a Protocol?](#1-what-is-a-protocol)
2. [The Problem — Why MCP Was Needed](#2-the-problem--why-mcp-was-needed)
3. [What Is MCP?](#3-what-is-mcp)
4. [Why Is MCP Overhyped? (And Why It's Still Important)](#4-why-is-mcp-overhyped-and-why-its-still-important)
5. [Basic Tool Calling vs MCP Tool Calling](#5-basic-tool-calling-vs-mcp-tool-calling)
6. [General Architecture of MCP](#6-general-architecture-of-mcp)
7. [MCP Components Deep Dive](#7-mcp-components-deep-dive)
8. [The Protocol Internals — STDIO & SSE](#8-the-protocol-internals--stdio--sse)
9. [MCP Handshake — How Client & Server Connect](#9-mcp-handshake--how-client--server-connect)
10. [MCP Folder Code Walkthrough](#10-mcp-folder-code-walkthrough)
11. [How to Run the MCP Server](#11-how-to-run-the-mcp-server)
12. [mcp.json — Connecting to Cursor / VS Code / Claude Desktop](#12-mcpjson--connecting-to-cursor--vs-code--claude-desktop)
13. [Adding MCP to an AI Project — HTTP Analogy](#13-adding-mcp-to-an-ai-project--http-analogy)
14. [Real-World MCP Examples](#14-real-world-mcp-examples)
15. [STDIO vs SSE — Limitations & Comparison](#15-stdio-vs-sse--limitations--comparison)
16. [Who Uses MCP? — Google, Anthropic, OpenAI](#16-who-uses-mcp--google-anthropic-openai)
17. [Quick Reference Cheat Sheet](#17-quick-reference-cheat-sheet)

---

## 1. What Is a Protocol?

A **protocol** is a **set of rules** that define how two systems communicate with each other.

### Why Should You Be Excited About "Just a Set of Rules"?

Think about it — **rules create standards, and standards create ecosystems**:

| Protocol | What It Standardized | Impact |
|----------|---------------------|--------|
| **HTTP** | How browsers talk to web servers | The entire web exists because of this |
| **TCP/IP** | How computers send data packets | The internet itself |
| **USB** | How peripherals connect to computers | One port for keyboard, mouse, drive, phone |
| **SMTP** | How email is sent | Gmail, Outlook, Yahoo — all interoperable |
| **MCP** | How AI apps talk to external tools | **Any AI can use any tool without custom code** |

Without HTTP, every browser would need custom code for every website. Without MCP, every AI app needs custom code for every tool. **That's the excitement** — MCP is doing for AI tools what HTTP did for the web.

### How HTTP Transfers Data (For Comparison)

```
HTTP → runs over TCP

┌──────────────────────────────────────┐
│           HTTP REQUEST               │
├──────────────────────────────────────┤
│  Method:  GET /api/weather           │  ← Header
│  Host:    api.example.com            │  ← Header
│  Accept:  application/json           │  ← Header
│                                      │
│  Body:    (empty for GET)            │  ← Body
└──────────────────────────────────────┘

                    ▼

┌──────────────────────────────────────┐
│           HTTP RESPONSE              │
├──────────────────────────────────────┤
│  Status:  200 OK                     │  ← Status Code
│  Content-Type: application/json      │  ← Header
│                                      │
│  Body:    {"temp": "25°C"}           │  ← Body (data)
└──────────────────────────────────────┘
```

MCP works similarly but uses **STDIO** (standard input/output) or **SSE** (Server-Sent Events) instead of raw TCP — more on this later.

---

## 2. The Problem — Why MCP Was Needed

### Problem 1: Every AI Platform Has Its Own Tool Format

Before MCP, if you wanted to give an AI model a tool, every platform had a **different format**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE CHAOS BEFORE MCP                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OpenAI:                                                            │
│  ┌──────────────────────────────────────────────┐                  │
│  │  tools = [{                                   │                  │
│  │    "type": "function",                        │                  │
│  │    "function": {                              │                  │
│  │      "name": "get_weather",                   │                  │
│  │      "parameters": { ... JSON Schema ... }    │                  │
│  │    }                                          │                  │
│  │  }]                                           │                  │
│  │  response.choices[0].message.tool_calls[0]    │   ← OpenAI way  │
│  │  response.tool_calls[0].function.name         │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  Anthropic (Claude):                                                │
│  ┌──────────────────────────────────────────────┐                  │
│  │  tools = [{                                   │                  │
│  │    "name": "get_weather",                     │                  │
│  │    "input_schema": { ... JSON Schema ... }    │                  │
│  │  }]                                           │                  │
│  │  response.content[0].input                    │   ← Claude way  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  LangChain:                                                         │
│  ┌──────────────────────────────────────────────┐                  │
│  │  @tool                                        │                  │
│  │  def get_weather(city: str):                  │                  │
│  │      """docstring = description"""            │   ← LangChain   │
│  │  llm.bind_tools([get_weather])                │      way        │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  Google Gemini:                                                     │
│  ┌──────────────────────────────────────────────┐                  │
│  │  Yet another format...                        │   ← Google way  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  Result: 4 different tools definitions for THE SAME tool!           │
└─────────────────────────────────────────────────────────────────────┘
```

### Problem 2: Tools Are Tightly Coupled to Applications

```
WITHOUT MCP:
                                                                 
  ┌──────────┐     def get_weather():         ┌──────────────┐   
  │  Your AI │────▶ fetch(api.weather.com)  ──▶│ Weather API  │   
  │  App     │     # hardcoded in YOUR code   └──────────────┘   
  └──────────┘                                                    
                                                                 
  ┌──────────┐     def get_weather():         ┌──────────────┐   
  │  Another │────▶ fetch(api.weather.com)  ──▶│ Weather API  │   
  │  AI App  │     # SAME code, written AGAIN └──────────────┘   
  └──────────┘                                                    
                                                                 
  ┌──────────┐     def get_weather():         ┌──────────────┐   
  │  Third   │────▶ fetch(api.weather.com)  ──▶│ Weather API  │   
  │  AI App  │     # SAME code, THIRD TIME    └──────────────┘   
  └──────────┘                                                    

  Every app re-implements the same tool. 10 apps × 10 tools = 100 integrations!
```

### Problem 3: No Standard for Tool Discovery

- In OpenAI, you define tools in the `tools` parameter → the **developer manually lists** them
- In system prompts, you describe tools in plain text → **LLM has to parse unstructured text**
- No way for an AI to **automatically discover** what tools are available

### Problem 4: The Side Project Scenario

```
Suppose you built a side project: Real-Time Stock Market Data API

WITHOUT MCP:
  1. You build the API (REST endpoints)
  2. Someone wants to use it in their AI app
  3. They have to:
     - Read your API docs
     - Write a def get_stock_price() function
     - Write prompt: "You have a tool called get_stock_price..."
     - Handle the response format manually
     - Do this AGAIN for every AI app that wants your data

WITH MCP:
  1. You build the API
  2. You ALSO create an MCP server that wraps your API
  3. Now ANY AI app (Claude, Cursor, your custom app) can:
     - Auto-discover your tools
     - Auto-understand parameters
     - Call them in a standard format
     - No custom integration code needed!
```

### Summary of Problems

| # | Problem | Without MCP | With MCP |
|---|---------|-------------|----------|
| 1 | Tool format | Every platform different | One standard format |
| 2 | Integration | N apps × M tools = N×M integrations | N + M (each connects to MCP) |
| 3 | Discovery | Manual tool listing | Auto-discovery (list_tools) |
| 4 | Reusability | Tools locked inside one app | Tools shareable across all apps |
| 5 | Context feeding | Manually stuff into prompts | Protocol handles it |

---

## 3. What Is MCP?

### Definition

**Model Context Protocol (MCP)** is an open standard created by **Anthropic** that provides a universal way for AI applications to connect with external data sources and tools.

> Think of it as a **USB-C port for AI** — one standard connector that works with everything.

### The Name Breakdown

```
Model    → The AI model (GPT, Claude, Gemini, Llama, etc.)
Context  → Data, tools, resources that the model needs
Protocol → Set of rules for how to exchange that context
```

**MCP = The rules for how to feed context (tools + data) into any AI model.**

### MCP in One Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   THE MCP ECOSYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   AI Apps (MCP Clients)          MCP Servers (Tool Providers)   │
│   ─────────────────────          ───────────────────────────    │
│                                                                 │
│   ┌───────────┐                  ┌─────────────────────┐       │
│   │  Claude   │─────────────────▶│  PostgreSQL MCP     │       │
│   │  Desktop  │                  │  (query databases)  │       │
│   └───────────┘\                 └─────────────────────┘       │
│                  \                                              │
│   ┌───────────┐  \               ┌─────────────────────┐       │
│   │  Cursor   │───\─────────────▶│  GitHub MCP         │       │
│   │  IDE      │    \             │  (repos, PRs, issues│       │
│   └───────────┘     \            └─────────────────────┘       │
│                      \                                          │
│   ┌───────────┐      \          ┌─────────────────────┐        │
│   │  Your     │───────\────────▶│  Gmail MCP          │        │
│   │  AI App   │        \        │  (read/send emails) │        │
│   └───────────┘         \       └─────────────────────┘        │
│                          \                                      │
│   ┌───────────┐           \     ┌─────────────────────┐        │
│   │  VS Code  │────────────\──▶│  Your Custom MCP    │        │
│   │  Copilot  │              \  │  (stock market data)│        │
│   └───────────┘               \ └─────────────────────┘        │
│                                \                                │
│                    All use the SAME protocol (MCP)!             │
│                    No custom integration needed!                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Why Is MCP Overhyped? (And Why It's Still Important)

### The Overhype

| Overhyped Claim | Reality |
|----------------|---------|
| "MCP will replace all APIs!" | No — MCP wraps APIs, it doesn't replace them. REST APIs still exist underneath. |
| "MCP is revolutionary technology!" | It's a **protocol standard** — the tech is JSON-RPC over STDIO/SSE. Not groundbreaking. |
| "Every AI tool MUST use MCP!" | Many tools work fine without it. MCP matters at **scale** (many tools, many apps). |
| "MCP is the only way!" | LangChain's `@tool`, OpenAI function calling — all work fine for single-app tools. |
| "MCP servers are cloud-hosted!" | Most run **locally** on your machine right now (STDIO mode). |

### Why It's STILL Important Despite the Hype

```
The value isn't in the technology — it's in the STANDARDIZATION.

Before USB:           After USB:
──────────            ─────────
- PS/2 for keyboard   - USB for everything
- Serial for mouse    - One port
- Parallel for printer - One cable
- FireWire for camera  - Plug and play
- Custom for each      

Before MCP:           After MCP:
──────────            ─────────
- OpenAI format        - MCP for everything
- Claude format        - One standard
- LangChain format     - Auto-discovery
- Custom for each      - Plug and play
```

**The protocol itself is simple. The ECOSYSTEM it creates is powerful.**

---

## 5. Basic Tool Calling vs MCP Tool Calling

### Approach 1: System Prompt (Oldest, Most Basic)

```python
# Tools described in plain text inside system prompt
system_prompt = """
You have access to the following tools:

1. get_weather(city: str) - Returns weather for a city
2. add(a: int, b: int) - Adds two numbers

When you want to use a tool, respond with:
TOOL_CALL: tool_name(arg1, arg2)
"""

# Problems:
# - LLM might not follow the format
# - You have to parse the response manually
# - No validation of arguments
# - No standard structure
```

### Approach 2: Platform-Specific Tool Calling (OpenAI, Claude)

```python
# OpenAI approach
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[...],
    tools=tools
)

# To use the tool call:
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name        # "get_weather"
arguments = json.loads(tool_call.function.arguments)  # {"city": "Delhi"}

# Now YOU have to:
if function_name == "get_weather":
    result = get_weather(arguments["city"])     # Call your own function
elif function_name == "add":
    result = add(arguments["a"], arguments["b"])
# ... manual dispatch for EVERY tool
```

**Problems with this approach:**
- Tool definitions are **verbose JSON** (lots of boilerplate)
- Tool execution is **manual** — you write if/elif chains
- Tools are **locked inside your app** — can't share them
- Access via `response.choices[0].message.tool_calls[0].function.name` — deeply nested, chaotic
- Format is OpenAI-specific — won't work with Claude or Gemini without rewriting

### Approach 3: LangChain @tool (Better, But Still Local)

```python
@tool
def get_weather(city: str):
    """Get weather for a city"""
    return fetch_weather(city)

llm.bind_tools([get_weather])
# Cleaner, but still locked to your app
```

### Approach 4: MCP (Universal Standard)

```javascript
// Define tool ONCE on an MCP server
server.tool(
  'get_weather',
  { city: z.string().describe('Name of the city') },
  async ({ city }) => {
    const response = await axios.get(`https://wttr.in/${city}?format=%C+%t`);
    return { content: [{ type: 'text', text: response.data }] };
  }
);
```

```
Now ANY MCP client can use this tool:
- Claude Desktop → auto-discovers get_weather → uses it
- Cursor IDE → auto-discovers get_weather → uses it
- Your custom app → auto-discovers get_weather → uses it
- VS Code Copilot → auto-discovers get_weather → uses it

NO code changes in any client. NO tool redefinition. ONE server, INFINITE clients.
```

### Comparison Table

| Aspect | System Prompt | OpenAI tools | LangChain @tool | **MCP** |
|--------|--------------|-------------|----------------|---------|
| Format | Plain text | JSON Schema | Python decorator | Standard protocol |
| Discovery | Manual | Manual | Manual | **Auto-discovery** |
| Validation | None | JSON Schema | Type hints | **Zod / JSON Schema** |
| Sharing | Not possible | Not possible | Not possible | **Any client can use** |
| Execution | Manual parse | Manual dispatch | Auto via ToolNode | **Auto via protocol** |
| Multi-app | ❌ | ❌ | ❌ | **✅** |

---

## 6. General Architecture of MCP

### The Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────────────────────────────────────┐               │
│   │              HOST APPLICATION              │               │
│   │         (Claude Desktop, Cursor,           │               │
│   │          VS Code, Your AI App)             │               │
│   │                                            │               │
│   │   ┌──────────────┐  ┌──────────────┐      │               │
│   │   │  MCP Client  │  │  MCP Client  │ ...  │               │
│   │   │  (one per     │  │  (one per     │      │               │
│   │   │   server)     │  │   server)     │      │               │
│   │   └──────┬───────┘  └──────┬───────┘      │               │
│   └──────────┼─────────────────┼───────────────┘               │
│              │                 │                                 │
│        MCP Protocol      MCP Protocol                           │
│        (STDIO/SSE)       (STDIO/SSE)                           │
│              │                 │                                 │
│   ┌──────────▼───────┐ ┌──────▼────────────┐                  │
│   │   MCP Server A   │ │   MCP Server B    │                  │
│   │                  │ │                    │                  │
│   │  ┌────────────┐  │ │  ┌────────────┐   │                  │
│   │  │  Tool 1    │  │ │  │  Tool 3    │   │                  │
│   │  │  Tool 2    │  │ │  │  Tool 4    │   │                  │
│   │  │  Resource 1│  │ │  │  Prompt 1  │   │                  │
│   │  └────────────┘  │ │  └────────────┘   │                  │
│   └──────────────────┘ └────────────────────┘                  │
│              │                 │                                 │
│        ┌─────▼─────┐    ┌─────▼──────┐                         │
│        │ Database  │    │  Web API   │     (Actual services)   │
│        │ File Sys  │    │  Cloud     │                         │
│        └───────────┘    └────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### HTTP Analogy — Understanding MCP Through HTTP

```
HTTP World:                              MCP World:
──────────                               ─────────

Browser (Client)                         AI App (Host + MCP Client)
    │                                        │
    │ HTTP Request                           │ MCP Request (JSON-RPC)
    │ GET /api/users                         │ tools/call { name: "add", args: {a:1,b:2} }
    ▼                                        ▼
Web Server                               MCP Server
    │                                        │
    │ Processes request                      │ Executes tool function
    │ Queries database                       │ Runs add(1, 2)
    ▼                                        ▼
HTTP Response                            MCP Response
    200 OK                                   { content: [{ type:"text", text:"3" }] }
    { "users": [...] }


  Client → sends request → Server          MCP Client → calls tool → MCP Server
  Server → processes     → Response         MCP Server → executes   → Result
  
  SAME CONCEPT. Different protocol.
```

### Key Difference: Multiple Servers

```
In HTTP:   Browser → ONE server at a time (different URLs)
In MCP:    AI App → MULTIPLE MCP servers SIMULTANEOUSLY

┌──────────┐     ┌─────────────────┐
│          │────▶│ Weather Server  │──▶ wttr.in API
│          │     └─────────────────┘
│  Claude  │     ┌─────────────────┐
│  Desktop │────▶│ GitHub Server   │──▶ GitHub API
│          │     └─────────────────┘
│          │     ┌─────────────────┐
│          │────▶│ Postgres Server │──▶ Your Database
└──────────┘     └─────────────────┘

Each server can have MULTIPLE tools.
The AI can use tools from ALL servers in a single conversation.
```

---

## 7. MCP Components Deep Dive

### The 3 Main Components

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP COMPONENTS                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. HOST                2. CLIENT              3. SERVER     │
│  ─────                  ──────                 ──────        │
│  The AI app             The connector          The provider  │
│  (Claude, Cursor,       (one per server,       (exposes      │
│  VS Code)               lives inside host)     tools/data)   │
│                                                              │
│  ┌────────────────────────────────────┐   ┌──────────────┐  │
│  │           HOST                     │   │              │  │
│  │  ┌────────────┐  ┌────────────┐   │   │  MCP Server  │  │
│  │  │ MCP Client │  │ MCP Client │   │   │  (separate   │  │
│  │  │     A      │  │     B      │   │   │   process)   │  │
│  │  └─────┬──────┘  └─────┬──────┘   │   │              │  │
│  └────────┼───────────────┼──────────┘   └──────────────┘  │
│           │               │                                  │
│     ┌─────▼──────┐  ┌─────▼──────┐                          │
│     │ MCP Server │  │ MCP Server │                          │
│     │     A      │  │     B      │                          │
│     └────────────┘  └────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

### What Each Component Does

| Component | Role | Example | Responsibility |
|-----------|------|---------|---------------|
| **Host** | The AI application | Claude Desktop, Cursor, VS Code | Manages clients, handles user interaction |
| **Client** | The connector | Auto-created by host (1 per server) | Maintains connection, sends requests, receives responses |
| **Server** | The tool provider | Your MCP server (index.js) | Exposes tools, resources, and prompts |

### What MCP Servers Can Expose (3 Primitives)

```
┌────────────────────────────────────────────────────────────┐
│              MCP SERVER PRIMITIVES                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. TOOLS                                                  │
│     ─────                                                  │
│     Functions the AI can call                              │
│     e.g., run_query(), get_weather(), send_email()         │
│     → Model-controlled (AI decides when to call)           │
│                                                            │
│  2. RESOURCES                                              │
│     ─────────                                              │
│     Data the AI can read (like GET endpoints)              │
│     e.g., file contents, database schemas, API responses   │
│     → Application-controlled (host decides when to load)   │
│                                                            │
│  3. PROMPTS                                                │
│     ───────                                                │
│     Pre-built prompt templates                             │
│     e.g., "Analyze this code", "Summarize this file"      │
│     → User-controlled (user picks from menu)              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### The Abstraction Layer

MCP provides a clean **abstraction layer** between AI apps and external services:

```
WITHOUT MCP (No Abstraction):
──────────────────────────────

AI App ──▶ knows API URL, auth tokens, request format, response parsing
           for EVERY service it talks to
           = tightly coupled, hard to maintain

WITH MCP (Abstraction):
───────────────────────

AI App ──▶ MCP Client ──▶ MCP Server ──▶ Actual Service
             │                │
             │ Doesn't know   │ Handles all the
             │ about the      │ API details,
             │ actual service  │ auth, parsing
             │                │
             └── ABSTRACTION ─┘

The AI app just says: "call tool X with args Y"
The MCP server handles everything else.
```

---

## 8. The Protocol Internals — STDIO & SSE

### What Is STDIO?

**STDIO** = **Standard Input/Output** — the most basic way programs communicate in any OS.

```
Every program has 3 standard streams:

┌───────────────────────────────────────────┐
│              ANY PROGRAM                  │
│                                           │
│   stdin  (standard input)   ← reads from │   ← keyboard / pipe
│   stdout (standard output)  → writes to  │   → terminal / pipe
│   stderr (standard error)   → errors to  │   → terminal / pipe
│                                           │
└───────────────────────────────────────────┘

Example in terminal:
$ echo "hello" | node index.js
         │              │
         │              └── reads from stdin
         └── writes to stdout, piped into next program's stdin
```

### How MCP Uses STDIO

```
┌──────────────────────────────────────────────────────────────┐
│              MCP OVER STDIO                                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  MCP Client (inside Host)         MCP Server (child process) │
│  ─────────────────────────        ─────────────────────────  │
│                                                              │
│  The Host SPAWNS the server                                  │
│  as a CHILD PROCESS                                          │
│  (e.g., runs: node index.js)                                 │
│                                                              │
│  ┌──────────┐   stdin    ┌──────────┐                       │
│  │          │──────────▶│          │                       │
│  │  Client  │           │  Server  │                       │
│  │          │◀──────────│ (node    │                       │
│  │          │   stdout   │  index.js)│                      │
│  └──────────┘           └──────────┘                       │
│                                                              │
│  Client writes JSON-RPC to server's stdin                    │
│  Server writes JSON-RPC to its stdout                        │
│  Server errors go to stderr (for logging)                    │
│                                                              │
│  ⚠️ Both run on the SAME MACHINE                            │
│  ⚠️ Cannot work over a NETWORK                              │
└──────────────────────────────────────────────────────────────┘
```

### How MCP Uses SSE (Server-Sent Events)

```
┌──────────────────────────────────────────────────────────────┐
│              MCP OVER SSE (HTTP-based)                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  MCP Client                    MCP Server                    │
│  (anywhere)                    (hosted on a server/cloud)    │
│                                                              │
│  ┌──────────┐   HTTP POST    ┌──────────┐                   │
│  │          │──────────────▶│          │                   │
│  │  Client  │               │  Server  │                   │
│  │          │◀──────────────│ (running │                   │
│  │          │   SSE stream   │  on port)│                   │
│  └──────────┘               └──────────┘                   │
│                                                              │
│  Client → POST to /message endpoint                          │
│  Server → Streams results back via SSE                       │
│                                                              │
│  ✅ Can work over a NETWORK                                  │
│  ✅ Can be hosted on a cloud server                          │
│  ✅ Multiple clients can connect                             │
└──────────────────────────────────────────────────────────────┘
```

### What Is SSE?

**SSE (Server-Sent Events)** is a web technology where the server can **push data to the client** over a single HTTP connection.

```
Normal HTTP:      Request → Response (done, connection closed)
SSE:              Request → Response stream... data... data... data... (keeps going)
```

Used in MCP to allow the server to stream tool results back to the client.

---

## 9. MCP Handshake — How Client & Server Connect

### The Connection Flow

```
┌──────────────────────────────────────────────────────────────────┐
│            MCP HANDSHAKE & TOOL DISCOVERY                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: HOST SPAWNS SERVER                                      │
│  ─────────────────────────                                       │
│                                                                  │
│  Host reads mcp.json → finds: "command": "node index.js"        │
│  Host spawns child process: node index.js                        │
│                                                                  │
│  ┌──────────┐   spawns    ┌──────────┐                          │
│  │   Host   │────────────▶│  Server  │                          │
│  │  (Cursor)│             │  Process │                          │
│  └──────────┘             └──────────┘                          │
│                                                                  │
│  Step 2: INITIALIZE (Client → Server)                            │
│  ────────────────────────────────────                            │
│                                                                  │
│  Client ──▶ { method: "initialize",                              │
│              params: {                                           │
│                clientInfo: { name: "Cursor", version: "1.0" },   │
│                capabilities: { ... }                             │
│              }}                                                   │
│                                                                  │
│  Server ──▶ { result: {                                          │
│                serverInfo: { name: "My Server", version: "1.0" },│
│                capabilities: { tools: {}, resources: {} }        │
│              }}                                                   │
│                                                                  │
│  Step 3: LIST TOOLS (Client → Server)                            │
│  ────────────────────────────────────                            │
│                                                                  │
│  Client ──▶ { method: "tools/list" }                             │
│                                                                  │
│  Server ──▶ { result: { tools: [                                 │
│                { name: "add",                                    │
│                  description: "Adds two numbers",                │
│                  inputSchema: { a: number, b: number } },        │
│                { name: "weather",                                │
│                  description: "Get weather for city",            │
│                  inputSchema: { city: string } }                 │
│              ]}}                                                  │
│                                                                  │
│  Now the client KNOWS all available tools and their schemas!     │
│                                                                  │
│  Step 4: CALL TOOL (Client → Server, when AI decides)            │
│  ────────────────────────────────────────────────────            │
│                                                                  │
│  User: "What's the weather in Delhi?"                            │
│  AI decides to use the weather tool                              │
│                                                                  │
│  Client ──▶ { method: "tools/call",                              │
│              params: {                                           │
│                name: "weather",                                  │
│                arguments: { city: "Delhi" }                      │
│              }}                                                   │
│                                                                  │
│  Server ──▶ { result: {                                          │
│                content: [{ type: "text", text: "Sunny +35°C" }]  │
│              }}                                                   │
│                                                                  │
│  AI uses this result to formulate its response to the user.      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Simplified Flow Diagram

```
Host starts → spawns server → initialize → tools/list → [wait for user]
                                                              │
                                              User asks something
                                                              │
                                    AI decides to call tool → tools/call
                                                              │
                                              Server executes tool
                                                              │
                                              Returns result to AI
                                                              │
                                    AI formulates response → shown to user
```

---

## 10. MCP Folder Code Walkthrough

> **Directory:** `MCP/`  
> **Files:** `index.js`, `package.json`, `package-lock.json`

### package.json

```json
{
  "name": "mcp-weather",               // Project name
  "version": "1.0.0",
  "main": "index.js",                  // Entry point
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.11.0",  // Official MCP SDK
    "axios": "^1.9.0",                       // HTTP client (for weather API)
    "zod": "^3.24.4"                         // Schema validation (defines tool params)
  }
}
```

| Dependency | Purpose |
|-----------|---------|
| `@modelcontextprotocol/sdk` | Official MCP SDK — provides `McpServer`, `StdioServerTransport`, etc. |
| `axios` | Makes HTTP requests (used to call the weather API) |
| `zod` | Schema validation library — defines and validates tool input parameters |

### index.js — Line-by-Line Explanation

```javascript
// ─────────────────── IMPORTS ───────────────────
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
// McpServer = the main class to create an MCP server

import { z } from 'zod';
// z = Zod schema builder (like JSON Schema but in JS)
// Used to define tool parameter types and validation

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
// StdioServerTransport = transport layer that uses STDIO (stdin/stdout)
// This means the server communicates via standard input/output

import axios from 'axios';
// HTTP client for making API calls
```

```javascript
// ─────────────────── CREATE SERVER ───────────────────
const server = new McpServer({
  name: 'My Server',       // Server name (shown during initialize handshake)
  version: '1.0.0',        // Server version
});
```

```javascript
// ─────────────────── TOOL 1: ADD ───────────────────
server.tool(
  'add',                              // Tool name
  { a: z.number(), b: z.number() },   // Input schema: two numbers
  async function ({ a, b }) {          // Handler function
    const sum = a + b;
    return {
      content: [{ type: 'text', text: String(sum) }]
      //                                    ▲
      //                                    │
      //  MCP requires content to be array of typed objects
      //  type: 'text' | 'image' | 'resource'
    };
  }
);
```

**How `server.tool()` works:**

```
server.tool(name, inputSchema, handler)
              │        │           │
              │        │           └── Async function that runs when tool is called
              │        └── Zod schema defining what arguments the tool accepts
              └── String name that the AI will use to call this tool

When a client calls tools/list → server returns:
  { name: "add", inputSchema: { a: number, b: number } }

When a client calls tools/call { name: "add", arguments: { a: 5, b: 3 } }:
  → handler runs → returns { content: [{ type: "text", text: "8" }] }
```

```javascript
// ─────────────────── TOOL 2: WEATHER ───────────────────
server.tool(
  'weather',
  { city: z.string().describe('Name of the city') },
  //                  ^^^^^^^^
  //  .describe() adds a description to the parameter
  //  This is sent to the AI so it knows what to pass
  
  async function ({ city }) {
    const response = await axios.get(
      `https://wttr.in/${city}?format=%C+%t`,
      //  wttr.in = free weather API
      //  %C = weather condition (Sunny, Cloudy, etc.)
      //  %t = temperature
      { responseType: 'json' }
    );
    return {
      content: [{ type: 'text', text: JSON.stringify(response.data) }]
    };
  }
);
```

```javascript
// ─────────────────── CONNECT TRANSPORT ───────────────────
const transport = new StdioServerTransport();
// Creates a STDIO transport (reads from stdin, writes to stdout)

await server.connect(transport);
// Connects the server to the transport
// Now the server is LISTENING for JSON-RPC messages on stdin
// and RESPONDING via stdout
```

### What Happens When `node index.js` Runs?

```
┌──────────────────────────────────────────────────────────┐
│           AFTER: node index.js                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. McpServer instance created                           │
│     - name: "My Server"                                  │
│     - version: "1.0.0"                                   │
│                                                          │
│  2. Tools registered:                                    │
│     - "add" → { a: number, b: number } → returns sum    │
│     - "weather" → { city: string } → returns weather     │
│                                                          │
│  3. StdioServerTransport created                         │
│     - Listens on process.stdin                           │
│     - Writes to process.stdout                           │
│                                                          │
│  4. server.connect(transport)                            │
│     - Server is now WAITING for JSON-RPC messages        │
│     - Blocks / stays alive                               │
│     - Ready to handle: initialize, tools/list, tools/call│
│                                                          │
│  ⚠️ Running manually = server just sits there waiting   │
│     You need an MCP CLIENT to send it commands!          │
│     (Claude Desktop, Cursor, etc.)                       │
└──────────────────────────────────────────────────────────┘
```

---

## 11. How to Run the MCP Server

### Step 1: Install Dependencies

```bash
cd MCP
npm install
```

### Step 2: Run the Server (Standalone Test)

```bash
node index.js
# Server starts and waits for input on stdin
# It won't print anything — it's waiting for JSON-RPC messages
# Press Ctrl+C to stop
```

> **Note:** Running `node index.js` alone isn't very useful — the server just waits for STDIO input. You need an MCP client (like Cursor or Claude Desktop) to connect to it.

### Step 3: Connect to an AI Tool

The real way to use an MCP server is through a **config file** that tells your AI tool how to start the server. See the next section.

---

## 12. mcp.json — Connecting to Cursor / VS Code / Claude Desktop

### What Is mcp.json?

A configuration file that tells your AI tool (Cursor, VS Code, Claude Desktop) **which MCP servers to start and how**.

### Location of Config Files

| AI Tool | Config File Location |
|---------|---------------------|
| **Cursor** | `.cursor/mcp.json` (in project root) or global settings |
| **VS Code (Copilot)** | `.vscode/mcp.json` (in project root) |
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) |

### mcp.json Format — Single Server

```json
{
  "mcpServers": {
    "my-weather-server": {
      "command": "node",
      "args": ["C:/path/to/MCP/index.js"],
      "env": {}
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `"my-weather-server"` | Name you give to this server (anything you want) |
| `"command"` | The executable to run (`node`, `python`, `npx`, etc.) |
| `"args"` | Arguments passed to the command (your server file path) |
| `"env"` | Environment variables to pass (API keys, etc.) |

### mcp.json — Multiple Servers

```json
{
  "mcpServers": {
    "weather-server": {
      "command": "node",
      "args": ["C:/path/to/MCP/index.js"]
    },
    "github-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    },
    "postgres-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/mydb"
      }
    }
  }
}
```

### What Happens When You Open Cursor with This Config

```
┌──────────────────────────────────────────────────────────────┐
│          CURSOR STARTUP WITH mcp.json                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Cursor reads .cursor/mcp.json                            │
│                                                              │
│  2. For EACH server in mcpServers:                           │
│     ┌─────────────────────────────────────────────────────┐  │
│     │ Spawns child process:                               │  │
│     │   node C:/path/to/MCP/index.js                      │  │
│     │   npx @modelcontextprotocol/server-github           │  │
│     │   npx @modelcontextprotocol/server-postgres         │  │
│     └─────────────────────────────────────────────────────┘  │
│                                                              │
│  3. Creates MCP Client for each server                       │
│                                                              │
│  4. Sends "initialize" to each server                        │
│                                                              │
│  5. Sends "tools/list" to each server                        │
│     → Gets list of all available tools                       │
│                                                              │
│  6. Tools are now available to the AI!                        │
│     User: "What's the weather in Delhi?"                     │
│     AI: Uses weather-server's "weather" tool                 │
│     User: "Show my GitHub issues"                            │
│     AI: Uses github-server's "list_issues" tool              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### For VS Code Copilot (`.vscode/mcp.json`)

```json
{
  "servers": {
    "my-weather-server": {
      "type": "stdio",
      "command": "node",
      "args": ["${workspaceFolder}/MCP/index.js"]
    }
  }
}
```

> VS Code uses `"servers"` instead of `"mcpServers"` and adds `"type": "stdio"`.

### SSE-Based Server Config (Remote/Hosted)

```json
{
  "mcpServers": {
    "remote-server": {
      "url": "http://your-server.com:3000/sse"
    }
  }
}
```

This connects to a server running **remotely** over HTTP+SSE instead of spawning a local process.

---

## 13. Adding MCP to an AI Project — HTTP Analogy

### Scenario: You Have an AI Project and Want to Add MCP

```
Your Current AI Project:
┌──────────────────────────────────────┐
│  main.py                             │
│  ├── LangGraph agent                 │
│  ├── @tool def run_command(cmd)      │  ← Tools defined INSIDE your app
│  ├── @tool def get_weather(city)     │  ← Can't be shared
│  └── graph.stream(...)               │
└──────────────────────────────────────┘

With MCP:
┌──────────────────────────────────────┐       ┌──────────────────┐
│  main.py                             │       │  MCP Server      │
│  ├── LangGraph agent                 │       │  (separate process)
│  ├── MCP Client connects to server ──┼──────▶│  ├── add()       │
│  ├── Tools auto-discovered           │       │  ├── weather()   │
│  └── graph.stream(...)               │       │  └── any tool... │
└──────────────────────────────────────┘       └──────────────────┘
```

### HTTP Reference Example

Think of how you'd add a REST API to your project:

```python
# WITHOUT external API (everything internal):
def get_weather(city):
    # Hardcoded logic inside your app
    return "Sunny 25°C"

# WITH external API (HTTP call):
import requests
def get_weather(city):
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()
```

MCP is the same concept but for AI tool calling:

```python
# WITHOUT MCP (tool defined inside your app):
@tool
def get_weather(city: str):
    return requests.get(f"https://wttr.in/{city}?format=%C+%t").text

# WITH MCP (tool lives on external MCP server):
# You don't define the tool at all!
# The MCP client auto-discovers it from the MCP server
# The AI calls it through the standard MCP protocol
```

---

## 14. Real-World MCP Examples

### Popular Community MCP Servers

| MCP Server | What It Does | Example Use |
|-----------|-------------|-------------|
| **@modelcontextprotocol/server-postgres** | Query PostgreSQL databases | "Show me all users who signed up this week" |
| **@modelcontextprotocol/server-github** | Interact with GitHub repos | "List open PRs on my repo" |
| **MCP Gmail** | Read/send emails | "Send an email to john@example.com" |
| **MCP Premiere Pro** | Control Adobe Premiere | "Split this video at 2:30" |
| **MCP Filesystem** | Read/write files on disk | "Show contents of /etc/hosts" |
| **MCP Brave Search** | Web search via Brave | "Search for latest Node.js release" |
| **MCP Puppeteer** | Browser automation | "Take a screenshot of google.com" |
| **MCP Slack** | Slack integration | "Send a message to #general" |

### Standardized Tools — The PostgreSQL Example

```
WITHOUT MCP:
  Every AI app that wants to query Postgres has to:
  1. Import psycopg2
  2. Write connection logic
  3. Write query execution logic
  4. Write result formatting logic
  5. Define tool schema for their specific LLM
  = EVERY app reimplements this

WITH MCP (postgres-server):
  1. Add to mcp.json: "@modelcontextprotocol/server-postgres"
  2. Set DATABASE_URL
  3. Done. ALL tools (query, list_tables, describe_table) auto-discovered.
  = Write ONCE, use in Claude, Cursor, VS Code, any MCP client
```

### Your Side Project Scenario — Stock Market Data

```
Step 1: You built a side project
┌─────────────────────────────────┐
│  Stock Market Data Service      │
│  ├── get_stock_price(ticker)    │
│  ├── get_market_summary()       │
│  └── get_history(ticker, days)  │
└─────────────────────────────────┘

Step 2: Create REST API for it (optional)
  GET /api/stock/AAPL → { price: 175.50 }

Step 3: Create MCP server wrapping your API
┌──────────────────────────────────────┐
│  stock-mcp-server/index.js           │
│                                      │
│  server.tool('get_stock_price',      │
│    { ticker: z.string() },           │
│    async ({ ticker }) => {           │
│      const data = await fetch(...)   │
│      return { content: [...] }       │
│    }                                 │
│  );                                  │
└──────────────────────────────────────┘

Step 4: Others add YOUR server to their mcp.json
{
  "mcpServers": {
    "sahil-stock-data": {
      "command": "node",
      "args": ["/path/to/stock-mcp-server/index.js"]
    }
  }
}

Step 5: Now ANY AI app can use your stock data!
  User in Claude: "What's Apple's stock price?"
  Claude → calls get_stock_price("AAPL") via MCP → "175.50"
  
  User in Cursor: "Show me TSLA history for 30 days"
  Cursor → calls get_history("TSLA", 30) via MCP → chart data

WITHOUT MCP:
  They'd have to:
  1. Read your API docs
  2. Write def get_stock_price() in THEIR codebase
  3. Add it to THEIR system prompt
  4. Handle response parsing
  = Custom integration for EVERY app
```

---

## 15. STDIO vs SSE — Limitations & Comparison

### Side-by-Side Comparison

| Feature | STDIO | SSE (Server-Sent Events) |
|---------|-------|--------------------------|
| **Transport** | stdin/stdout pipes | HTTP + SSE streaming |
| **Where server runs** | **Local only** (same machine) | **Local or Remote** (any server) |
| **How it starts** | Host spawns child process | Server runs independently |
| **Network support** | ❌ Cannot work over network | ✅ Works over HTTP |
| **Multiple clients** | ❌ One client per process | ✅ Multiple clients |
| **Hosting** | ❌ Can't host on cloud | ✅ Can host on cloud |
| **Config** | `"command": "node", "args": [...]` | `"url": "http://..."` |
| **Setup complexity** | Simple (just a command) | More complex (HTTP server) |
| **Current status** | Default / most common | Growing adoption |
| **Best for** | Local dev tools, IDE plugins | Production, shared services |

### STDIO Limitation: "Main server par host nahi kar sakta"

```
STDIO Problem:
  ┌────────┐    stdin/stdout     ┌──────────┐
  │ Client │◀───────────────────▶│  Server  │
  │        │   (Process pipes)   │  (child) │
  └────────┘                     └──────────┘
        │                              │
        └──── SAME MACHINE ONLY ───────┘
        
  ❌ Cannot share server with others
  ❌ Cannot host on AWS/GCP/cloud
  ❌ Each client spawns its own server process
  ❌ No URL to point to
```

```
SSE Solution:
  ┌────────┐                     ┌──────────┐
  │Client 1│──── HTTP + SSE ────▶│          │
  └────────┘                     │  MCP     │
  ┌────────┐                     │  Server  │──▶ Cloud/VPS
  │Client 2│──── HTTP + SSE ────▶│  (hosted)│
  └────────┘                     │          │
  ┌────────┐                     │          │
  │Client 3│──── HTTP + SSE ────▶│          │
  └────────┘                     └──────────┘
  
  ✅ Host on any server
  ✅ Multiple clients connect
  ✅ Share via URL
```

### SSE Config Example

```json
// mcp.json for SSE (remote server)
{
  "mcpServers": {
    "remote-stock-server": {
      "url": "https://my-mcp-server.onrender.com/sse"
    }
  }
}
```

### How SSE Helps with Code

```javascript
// Converting STDIO server → SSE server

// STDIO version (current index.js):
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
const transport = new StdioServerTransport();
await server.connect(transport);

// SSE version (hostable):
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import express from 'express';

const app = express();
app.get('/sse', async (req, res) => {
  const transport = new SSEServerTransport('/message', res);
  await server.connect(transport);
});
app.post('/message', async (req, res) => {
  // Handle incoming messages
});
app.listen(3000);  // Now accessible at http://localhost:3000/sse
```

> **Newer alternative:** MCP also supports **Streamable HTTP** transport which is an evolution beyond SSE, using standard HTTP POST with optional streaming.

---

## 16. Who Uses MCP? — Google, Anthropic, OpenAI

### MCP Adoption

| Company | Role | How They Use MCP |
|---------|------|-----------------|
| **Anthropic** | **Creator & Owner** | Created MCP. Claude Desktop has native MCP client support. |
| **Google** | Adopter | Announced MCP support in Gemini and Google AI Studio |
| **OpenAI** | Adopter | ChatGPT and Agents SDK support MCP |
| **Microsoft** | Adopter | VS Code Copilot, Copilot Studio support MCP |
| **Cursor** | Early adopter | One of the first IDEs with full MCP support |
| **Windsurf** | Adopter | MCP support in their AI IDE |
| **Sourcegraph** | Adopter | Cody supports MCP servers |
| **Block (Square)** | Adopter | Using MCP in their AI tools |
| **Apollo** | Adopter | Using MCP for their AI integrations |

### The Industry Move

```
2024 Nov:  Anthropic releases MCP as open standard
2025:      Google, OpenAI, Microsoft announce MCP support
2026:      MCP becoming the de-facto standard

Key Point:
   Anthropic CREATED it, but it's OPEN SOURCE.
   Even competitors (OpenAI, Google) adopted it because
   the standard benefits everyone.
   
   Think of it like: Google created Android, but Samsung,
   Xiaomi, OnePlus all use it. The standard benefits the ecosystem.
```

---

## 17. Quick Reference Cheat Sheet

### Creating an MCP Server (Node.js)

```javascript
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const server = new McpServer({ name: 'server-name', version: '1.0.0' });

// Define a tool
server.tool(
  'tool_name',                                    // Name
  { param: z.string().describe('description') },  // Input schema (Zod)
  async ({ param }) => {                           // Handler
    return { content: [{ type: 'text', text: 'result' }] };
  }
);

// Connect via STDIO
const transport = new StdioServerTransport();
await server.connect(transport);
```

### package.json for MCP Server

```json
{
  "type": "module",
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.11.0",
    "zod": "^3.24.0"
  }
}
```

### mcp.json Template

```json
{
  "mcpServers": {
    "server-name": {
      "command": "node",
      "args": ["path/to/index.js"],
      "env": { "API_KEY": "xxx" }
    }
  }
}
```

### MCP Message Flow

```
Client → initialize          → Server (handshake)
Client → tools/list          → Server (discover tools)
Client → tools/call {name, args} → Server (execute tool)
Server → { content: [...] }  → Client (return result)
```

### Key MCP Concepts

| Concept | One-Liner |
|---------|-----------|
| Host | The AI application (Claude, Cursor, VS Code) |
| Client | Connector inside host (one per server) |
| Server | Exposes tools, resources, prompts |
| Tools | Functions AI can call (model-controlled) |
| Resources | Data AI can read (app-controlled) |
| Prompts | Pre-built templates (user-controlled) |
| STDIO | Local transport (stdin/stdout) |
| SSE | Remote transport (HTTP streaming) |
| JSON-RPC | Message format used by MCP protocol |

---

> **Summary:** MCP is Anthropic's open protocol that standardizes how AI apps connect to tools. Before MCP, every platform (OpenAI, Claude, LangChain) had its own tool format — leading to N×M integration chaos. MCP solves this with a universal standard: define a tool ONCE on an MCP server, and ANY MCP client (Claude, Cursor, VS Code) can auto-discover and use it. The protocol runs over STDIO (local) or SSE (remote), uses JSON-RPC for messages, and follows an initialize → list_tools → call_tool flow. It's overhyped because the tech is simple (JSON over pipes), but it's genuinely important because **standardization creates ecosystems** — just like HTTP created the web.