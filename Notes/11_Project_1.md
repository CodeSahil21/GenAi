# Project 1 — Voice-Controlled AI Coding Assistant (Lang_Project)

> A real-time voice-driven coding assistant built with **LangGraph**, **SpeechRecognition**, **OpenAI GPT-4.1**, and **MongoDB checkpointing**.  
> The user speaks into a microphone → speech is transcribed → sent to an AI agent that can execute shell commands → response printed.

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Directory Structure](#2-directory-structure)
3. [Complete Workflow Diagram](#3-complete-workflow-diagram)
4. [SpeechRecognition Library — Deep Dive](#4-speechrecognition-library--deep-dive)
5. [graph.py — Line-by-Line Walkthrough](#5-graphpy--line-by-line-walkthrough)
6. [main.py — Line-by-Line Walkthrough](#6-mainpy--line-by-line-walkthrough)
7. [LangGraph Concepts Used in This Project](#7-langgraph-concepts-used-in-this-project)
8. [MongoDB Checkpointing & Docker Compose](#8-mongodb-checkpointing--docker-compose)
9. [OpenAI Text-to-Speech (TTS)](#9-openai-text-to-speech-tts)
10. [requirements.txt Breakdown](#10-requirementstxt-breakdown)
11. [How to Run This Project](#11-how-to-run-this-project)
12. [Potential Improvements](#12-potential-improvements)
13. [Quick Reference Cheat Sheet](#13-quick-reference-cheat-sheet)

---

## 1. Project Overview & Architecture

### What does this project do?

This is a **voice-controlled AI coding assistant**. Instead of typing commands, you **speak** into your microphone. The system:

1. **Listens** to your voice via the microphone
2. **Transcribes** your speech to text using Google's Speech Recognition API
3. **Sends** the text to a LangGraph-powered AI agent (GPT-4.1)
4. **Executes** shell commands if the AI decides a command needs to run (e.g., `mkdir`, `ls`, `python script.py`)
5. **Returns** the response and prints it to the terminal
6. **Remembers** conversation history via MongoDB checkpointing

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VOICE-CONTROLLED AI ASSISTANT                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│   │          │    │  Speech      │    │  LangGraph   │    │  Tool    │ │
│   │ 🎤 Mic   │───▶│  Recognition │───▶│  Chatbot     │───▶│ Executor │ │
│   │          │    │  (Google)    │    │  (GPT-4.1)   │    │ (Shell)  │ │
│   └──────────┘    └──────────────┘    └──────┬───────┘    └────┬─────┘ │
│                                              │                  │       │
│                                              │    ┌─────────────┘       │
│                                              ▼    ▼                     │
│                                        ┌───────────────┐               │
│                                        │  MongoDB      │               │
│                                        │  Checkpointer │               │
│                                        │  (Memory)     │               │
│                                        └───────────────┘               │
│                                                                         │
│   [Optional] ┌──────────────┐                                          │
│              │  OpenAI TTS  │  (gpt-4o-mini-tts, voice: "coral")       │
│              │  speak()     │  — defined but not used in main loop      │
│              └──────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | OpenAI GPT-4.1 | Brain of the coding assistant |
| Orchestration | LangGraph (StateGraph) | Agent loop with conditional tool calling |
| Speech-to-Text | SpeechRecognition + Google API | Convert voice to text |
| Text-to-Speech | OpenAI gpt-4o-mini-tts | Convert text to voice (optional) |
| Memory | MongoDB + langgraph-checkpoint-mongodb | Persist conversation across runs |
| Infrastructure | Docker Compose | Run MongoDB container |
| Tool | os.system() | Execute shell commands on user's machine |

---

## 2. Directory Structure

```
Lang_Project/
├── app/
│   ├── graph.py          # LangGraph definition — State, tools, nodes, edges
│   └── main.py           # Entry point — speech recognition + graph execution
├── chat_gpt/             # Output folder — AI saves generated files here
├── docker-compose.yml    # MongoDB container configuration
└── requirements.txt      # Python dependencies
```

### What Each File Does

| File | Lines | Role |
|------|-------|------|
| `app/graph.py` | 63 | Defines the LangGraph: State schema, `run_command` tool, chatbot node, conditional edges |
| `app/main.py` | 53 | Main loop: listen to mic → transcribe → stream through graph → print response |
| `docker-compose.yml` | 15 | Spins up a MongoDB container for checkpointing |
| `requirements.txt` | 5 | Lists extra dependencies (SpeechRecognition, audioop-lts, etc.) |
| `chat_gpt/` | — | Empty folder where the AI assistant saves generated code files |

---

## 3. Complete Workflow Diagram

### Step-by-Step Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE EXECUTION FLOW                          │
└──────────────────────────────────────────────────────────────────────────┘

  User Speaks              System Listens            Google API
  ───────────              ──────────────            ──────────
       │                        │                        │
       │    "Create a hello     │                        │
       │     world Python       │                        │
       │     script"            │                        │
       │                        │                        │
       ▼                        ▼                        ▼
  ┌─────────┐           ┌──────────────┐         ┌──────────────┐
  │  User   │──voice───▶│ sr.Microphone│──audio──▶│ recognize_   │
  │  speaks │           │ sr.listen()  │         │ google()     │
  └─────────┘           └──────────────┘         └──────┬───────┘
                                                        │
                                                   text string
                                                        │
                                                        ▼
                          ┌─────────────────────────────────────────┐
                          │           LANGGRAPH EXECUTION           │
                          ├─────────────────────────────────────────┤
                          │                                         │
                          │  ┌─────────┐     ┌──────────────────┐  │
                          │  │  START  │────▶│    chatbot node   │  │
                          │  └─────────┘     │  (GPT-4.1 +      │  │
                          │                  │   SystemMessage)  │  │
                          │                  └────────┬─────────┘  │
                          │                           │            │
                          │                  tools_condition()     │
                          │                     /         \        │
                          │                    /           \       │
                          │            has tool_calls    no calls  │
                          │                  /               \     │
                          │                 ▼                 ▼    │
                          │          ┌──────────┐       ┌───────┐ │
                          │          │  tools   │       │  END  │ │
                          │          │  (ToolNode)│      └───────┘ │
                          │          │ run_command│                │
                          │          └─────┬────┘                 │
                          │                │                       │
                          │                └──────▶ chatbot ──▶ …  │
                          │              (loop until no more calls) │
                          └─────────────────────────────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │  pretty_print()  │
                                        │  (Terminal Output)│
                                        └──────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │  Loop back to    │
                                        │  "Say something!"│
                                        └──────────────────┘
```

### Concrete Example Run

```
User says:    "Create a Python file that prints hello world"
                    │
                    ▼
Google STT:   "Create a Python file that prints hello world"
                    │
                    ▼
GPT-4.1:      Decides to call run_command tool with:
              cmd = 'echo "print(\"hello world\")" > chat_gpt/hello.py'
                    │
                    ▼
ToolNode:     Executes os.system('echo "print(\"hello world\")" > chat_gpt/hello.py')
              Returns: 0 (success)
                    │
                    ▼
GPT-4.1:      "I've created a Python file at chat_gpt/hello.py that prints hello world."
                    │
                    ▼
Terminal:     AI: I've created a Python file at chat_gpt/hello.py that prints hello world.
                    │
                    ▼
              "Say something!" (waiting for next voice input)
```

---

## 4. SpeechRecognition Library — Deep Dive

> **Source:** [PyPI](https://pypi.org/project/SpeechRecognition/) | [GitHub: Uberi/speech_recognition](https://github.com/Uberi/speech_recognition) | 9k+ stars | BSD-3-Clause License

### What Is It?

**SpeechRecognition** is a Python library for performing speech recognition, with support for **several engines and APIs**, both online and offline. It acts as a **unified wrapper** — you write your code once and can switch between recognition backends easily.

- **Install:** `pip install SpeechRecognition`
- **Quick test:** `python -m speech_recognition` (runs a demo after install)
- **Import as:** `import speech_recognition as sr`
- **Current Version:** 3.14.5 (latest on PyPI as of Dec 2025)
- **Python Requirement:** 3.9+

### Supported Speech Recognition Engines

| Engine | Online/Offline | API Key Needed? | Method |
|--------|---------------|-----------------|--------|
| **Google Speech Recognition** | Online | No (free, limited) | `recognize_google()` |
| Google Cloud Speech API | Online | Yes | `recognize_google_cloud()` |
| CMU Sphinx (PocketSphinx) | **Offline** | No | `recognize_sphinx()` |
| OpenAI Whisper (local) | **Offline** | No | `recognize_whisper()` |
| OpenAI Whisper API | Online | Yes | `recognize_openai()` |
| Groq Whisper API | Online | Yes | `recognize_groq()` |
| Faster Whisper | **Offline** | No | `recognize_faster_whisper()` |
| Vosk API | **Offline** | No | `recognize_vosk()` |
| Microsoft Azure Speech | Online | Yes | `recognize_azure()` |
| Wit.ai | Online | Yes | `recognize_wit()` |
| Houndify API | Online | Yes | `recognize_houndify()` |
| IBM Speech to Text | Online | Yes | `recognize_ibm()` |
| Tensorflow | **Offline** | No | — |

> **This project uses `recognize_google()`** — the free Google Speech Recognition API that requires **no API key**. It's the simplest to use but has rate limits for heavy usage.

### Key Classes & Methods Used in This Project

```python
import speech_recognition as sr

# 1. Recognizer — the main class
r = sr.Recognizer()

# 2. Microphone — captures audio from system microphone
# Requires PyAudio: pip install SpeechRecognition[audio]
with sr.Microphone() as source:
    
    # 3. adjust_for_ambient_noise — calibrates for background noise
    #    Listens for ~1 second to gauge ambient noise level
    #    Sets energy_threshold automatically
    r.adjust_for_ambient_noise(source)
    
    # 4. pause_threshold — seconds of silence before phrase is considered complete
    #    Default: 0.8 seconds. This project uses 2 seconds (longer pauses allowed)
    r.pause_threshold = 2

    # 5. listen — blocks until speech is detected, then records until silence
    #    Returns an AudioData object
    audio = r.listen(source)

    # 6. recognize_google — sends audio to Google's free STT API
    #    Returns: string of transcribed text
    #    Raises: sr.UnknownValueError if speech not understood
    #    Raises: sr.RequestError if API is unreachable
    text = r.recognize_google(audio)
```

### How `Recognizer` Works Internally

```
                        Recognizer
                    ┌─────────────────┐
                    │                 │
  audio stream ───▶│  energy_threshold│──▶ "Is this speech or noise?"
                    │  (auto-set by   │
                    │  adjust_for_    │
                    │  ambient_noise) │
                    │                 │
                    │  pause_threshold│──▶ "Has the person stopped talking?"
                    │  (2 seconds)    │     (2s of silence = end of phrase)
                    │                 │
                    │  listen()       │──▶ Returns AudioData object
                    │                 │
                    │  recognize_*()  │──▶ Sends AudioData to engine
                    │                 │     Returns transcribed text
                    └─────────────────┘
```

### Important Recognizer Properties

| Property | Default | Description |
|----------|---------|-------------|
| `energy_threshold` | 300 | Minimum audio energy to consider speech. Auto-adjusted by `adjust_for_ambient_noise()` |
| `dynamic_energy_threshold` | True | Automatically adjusts energy threshold over time |
| `pause_threshold` | 0.8 | Seconds of silence after speech to consider phrase complete |
| `phrase_threshold` | 0.3 | Minimum seconds of speech to consider valid |
| `non_speaking_duration` | 0.5 | Seconds of silence to keep before phrase |

### PyAudio Dependency

`sr.Microphone` requires **PyAudio** to access the system microphone:
- **Windows:** `pip install SpeechRecognition[audio]`
- **Linux (Debian):** `sudo apt-get install python-pyaudio python3-pyaudio`
- **macOS:** `brew install portaudio && pip install SpeechRecognition[audio]`

Without PyAudio, everything else in the library works (file-based transcription, etc.), but `Microphone()` will raise an `AttributeError`.

### Listing Available Microphones

```python
import speech_recognition as sr
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone(device_index={index}): {name}")
```

Use `sr.Microphone(device_index=N)` to select a specific microphone instead of the system default.

---

## 5. graph.py — Line-by-Line Walkthrough

> **File:** `Lang_Project/app/graph.py` (63 lines)  
> **Purpose:** Defines the LangGraph agent — State, tool, chatbot node, and graph wiring.

### Full Code with Annotations

```python
# ─────────────────── IMPORTS ───────────────────
import os                                                # Line 1: For os.system() shell execution
from typing import Annotated                             # Line 2: For annotating State fields
from typing_extensions import TypedDict                  # Line 3: For defining State schema
from langgraph.graph.message import add_messages         # Line 4: Reducer — appends messages to list
from langchain_openai import init_chat_model             # Line 5: Initialize any chat model
from langchain_core.messages import SystemMessage        # Line 6: System prompt message type
from langgraph.prebuilt import ToolNode, tools_condition # Line 7: Prebuilt tool execution components
from langgraph.graph import StateGraph, START, END       # Line 8: Graph building blocks
from langchain_core.tools import tool                    # Line 9: @tool decorator


# ─────────────────── STATE DEFINITION ───────────────────
class State(TypedDict):                                  # Line 12: TypedDict = schema for graph state
    messages: Annotated[list, add_messages]               # Line 13: Messages list with add_messages reducer
```

**What `add_messages` does:**
- Without `add_messages`: returning `{"messages": [new_msg]}` would **replace** the entire list
- With `add_messages`: returning `{"messages": [new_msg]}` **appends** the new message to existing ones
- This is how conversation history is preserved across node invocations

```python
# ─────────────────── TOOL DEFINITION ───────────────────
@tool                                                    # Line 15: Converts function to a LangChain Tool
def run_command(cmd: str):                               # Line 16: Takes a shell command string
    """                                                  # Line 17-21: Docstring = tool description for LLM
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    result = os.system(command=cmd)                      # Line 22: Execute command in system shell
    return result                                        # Line 23: Returns exit code (0 = success)
```

**Why `@tool` is important:**
- The `@tool` decorator converts a regular Python function into a LangChain `Tool` object
- The **docstring becomes the tool description** that the LLM reads to decide when to use it
- The **function parameters** (with type hints) become the tool's input schema
- The LLM generates a `tool_call` with `{"cmd": "ls -la"}` → LangGraph's ToolNode invokes it

**Security Note:** `os.system()` executes arbitrary shell commands. This is intentional for a local coding assistant but would be dangerous in production/web-facing apps.

```python
# ─────────────────── LLM SETUP ───────────────────
llm = init_chat_model(                                   # Line 25-27: Create a chat model instance
    model_provider="openai", model="gpt-4.1"
)
llm_with_tool = llm.bind_tools(tools=[run_command])      # Line 28: Bind the tool so LLM knows about it
```

**`init_chat_model` vs `ChatOpenAI`:**
- `init_chat_model` is a **universal factory** — can create models from any provider (openai, anthropic, google, etc.)
- `bind_tools()` tells the LLM about available tools and their schemas
- After binding, the LLM can generate `tool_calls` in its response when it decides to use a tool

```python
# ─────────────────── CHATBOT NODE ───────────────────
def chatbot(state: State):                               # Line 30: Node function — receives full state
    system_prompt = SystemMessage(content="""             # Line 31-38: System prompt defining AI behavior
        You are an AI Coding assistant who takes an input from user and based on available
        tools you choose the correct tool and execute the commands.
                                  
        You can even execute commands and help user with the output of the command.

        Always make sure to keep your generated codes and files in chat_gpt/ folder. 
        you can create one if not already there.                           
    """)

    message = llm_with_tool.invoke(                      # Line 40: Call LLM with system + conversation
        [system_prompt] + state["messages"]               #         Prepends system prompt every time
    )
    return {"messages": [message]}                        # Line 42: Return new message (add_messages appends it)
```

**Key Points:**
- `SystemMessage` is prepended to the message list on **every invocation** — it's not stored in state
- The node returns `{"messages": [message]}` — the `add_messages` reducer handles appending
- If the LLM decides to call a tool, `message.tool_calls` will contain the tool call details
- If no tool is needed, `message.content` will contain the text response

```python
# ─────────────────── TOOL NODE ───────────────────
tool_node = ToolNode(tools=[run_command])                # Line 44: Prebuilt node that executes tool calls
```

**`ToolNode`** is a LangGraph prebuilt that:
1. Reads the last message from state
2. Extracts any `tool_calls` from it
3. Executes the corresponding tool functions
4. Returns `ToolMessage` results back to state

```python
# ─────────────────── GRAPH BUILDING ───────────────────
graph_builder = StateGraph(State)                        # Line 46: Create graph with State schema

graph_builder.add_node("chatbot", chatbot)               # Line 48: Register chatbot node
graph_builder.add_node("tools", tool_node)               # Line 49: Register tools node

graph_builder.add_edge(START, "chatbot")                 # Line 51: START → chatbot (always)
graph_builder.add_conditional_edges(                     # Line 52-54: Conditional routing from chatbot
    "chatbot",
    tools_condition,                                     #   tools_condition checks for tool_calls
)
graph_builder.add_edge("tools", "chatbot")               # Line 56: tools → chatbot (always loop back)
graph_builder.add_edge("chatbot", END)                   # Line 57: chatbot → END (when no tool calls)
```

### Graph Visualization

```
                    ┌─────────────────────────────────┐
                    │         LANGGRAPH AGENT          │
                    └─────────────────────────────────┘

                              ┌───────┐
                              │ START │
                              └───┬───┘
                                  │
                                  ▼
                          ┌───────────────┐
                     ┌───▶│   chatbot     │◀──────┐
                     │    │  (GPT-4.1)    │       │
                     │    └───────┬───────┘       │
                     │            │               │
                     │     tools_condition()      │
                     │        /       \           │
                     │       /         \          │
                     │  tool_calls   no calls     │
                     │     /             \        │
                     │    ▼               ▼       │
                     │ ┌────────┐    ┌────────┐   │
                     │ │ tools  │    │  END   │   │
                     │ │ToolNode│    └────────┘   │
                     └─┤run_cmd │                 │
                       └────────┘                 │
```

**`tools_condition`** is a prebuilt function that:
- Checks if the last message has `tool_calls`
- If **yes** → route to `"tools"` node
- If **no** → route to `END`

```python
# ─────────────────── COMPILATION ───────────────────
graph = graph_builder.compile()                          # Line 59: Compile without checkpointer (no memory)

def create_chat_graph(checkpointer):                     # Line 61-62: Factory function for memory-enabled graph
    return graph_builder.compile(checkpointer=checkpointer)
```

**Two compilation modes:**
| Variable | Checkpointer | Memory | Use Case |
|----------|-------------|--------|----------|
| `graph` | None | No memory | Quick one-off queries |
| `create_chat_graph(cp)` | Yes | Persists across calls | Main app uses this with MongoDB |

---

## 6. main.py — Line-by-Line Walkthrough

> **File:** `Lang_Project/app/main.py` (53 lines)  
> **Purpose:** Entry point — sets up speech recognition, MongoDB checkpointing, and runs the main loop.

### Full Code with Annotations

```python
# ─────────────────── IMPORTS ───────────────────
from dotenv import load_dotenv                           # Line 1: Load .env file (OPENAI_API_KEY, etc.)
import speech_recognition as sr                          # Line 2: SpeechRecognition library
from langgraph.checkpoint.mongodb import MongoDBSaver    # Line 3: MongoDB checkpointer for LangGraph
from .graph import create_chat_graph                     # Line 4: Import graph factory (relative import)
import asyncio                                           # Line 5: For async TTS function
from openai.helpers import LocalAudioPlayer              # Line 6: Play audio locally (for TTS)
from openai import AsyncOpenAI                           # Line 7: Async OpenAI client (for TTS)

load_dotenv()                                            # Line 9: Load environment variables from .env

openai = AsyncOpenAI()                                   # Line 11: Create async OpenAI client
```

```python
# ─────────────────── CONFIGURATION ───────────────────
MONGODB_URI = "mongodb://admin:admin@localhost:27017"     # Line 13: MongoDB connection string
config = {"configurable": {"thread_id": "7"}}            # Line 14: Thread ID for checkpointing
```

**`thread_id` explanation:**
- Every conversation in LangGraph checkpointing is identified by a `thread_id`
- Same `thread_id` = same conversation (messages persist across runs)
- Different `thread_id` = new conversation (clean slate)
- Here it's hardcoded to `"7"` — all runs share the same conversation

```python
# ─────────────────── MAIN FUNCTION ───────────────────
def main():                                              # Line 17
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:  # Line 18: Context manager for MongoDB
        graph = create_chat_graph(checkpointer=checkpointer)           # Line 19: Compile graph with memory
```

**`MongoDBSaver.from_conn_string()`:**
- Creates a checkpointer that saves/loads state from MongoDB
- Used as context manager (`with`) to ensure proper cleanup
- Stores full graph state (all messages) for each thread_id
- Enables **conversation persistence** — restart the script and the AI remembers previous messages

```python
        r = sr.Recognizer()                              # Line 22: Create speech recognizer

        with sr.Microphone() as source:                  # Line 24: Open microphone (context manager)
            r.adjust_for_ambient_noise(source)           # Line 25: Calibrate for background noise (~1s)
            r.pause_threshold = 2                        # Line 26: Wait 2s of silence before ending phrase
```

| Setting | Value | Why |
|---------|-------|-----|
| `adjust_for_ambient_noise(source)` | Auto | Listens to ambient noise for ~1 second, sets `energy_threshold` |
| `pause_threshold = 2` | 2 seconds | Lets user pause mid-sentence without cutting off (default 0.8s is too short for coding instructions) |

```python
            while True:                                  # Line 28: Infinite loop — keeps listening
                print("Say something!")                  # Line 29: Prompt user
                audio = r.listen(source)                 # Line 30: BLOCKS until speech detected + silence

                print("Processing audio...")             # Line 32
                sst = r.recognize_google(audio)          # Line 33: Send audio to Google → get text

                print("You Said:", sst)                  # Line 35: Show transcription
                for event in graph.stream(               # Line 36-37: Stream graph execution
                    {"messages": [{"role": "user", "content": sst}]},  # User message
                    config,                              #             Thread config for checkpointing
                    stream_mode="values"                 #             Stream full state at each step
                ):
                    if "messages" in event:              # Line 39: Check if messages exist
                        event["messages"][-1].pretty_print()  # Line 40: Print latest message
```

**`graph.stream()` explained:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| First arg | `{"messages": [...]}` | Input to the graph (user's transcribed speech) |
| `config` | `{"configurable": {"thread_id": "7"}}` | Which conversation thread to use |
| `stream_mode` | `"values"` | Emit full state after each node execution |

**`stream_mode="values"` output:**
- After `chatbot` runs → emits full state (all messages including AI response)
- After `tools` runs → emits full state (all messages including tool result)
- After `chatbot` runs again → emits full state (all messages including final response)

**`event["messages"][-1].pretty_print()`** prints the **last message** in the conversation, which is the most recent AI response or tool result.

```python
# ─────────────────── TEXT-TO-SPEECH (OPTIONAL) ───────────────────
async def speak(text: str):                              # Line 43: Async TTS function
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",                         # Line 45: OpenAI TTS model
        voice="coral",                                   # Line 46: Voice preset
        input=text,                                      # Line 47: Text to speak
        instructions="Speak in a cheerful and positive tone.",  # Line 48: Voice style
        response_format="pcm",                           # Line 49: Raw PCM audio format
    ) as response:
        await LocalAudioPlayer().play(response)          # Line 51: Play audio through speakers

main()                                                   # Line 53: Run the main function

# if __name__ == "__main__":                             # Line 55-56: Commented out TTS demo
#      asyncio.run(speak(text="This is a sample voice. Hi Piyush"))
```

**The `speak()` function is defined but NOT called in the main loop.** It could be integrated to make the assistant speak its responses aloud.

---

## 7. LangGraph Concepts Used in This Project

### Concept Map

```
┌──────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH CONCEPTS IN PROJECT                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐ │
│  │   State     │    │   Nodes    │    │   Edges               │ │
│  │ (TypedDict) │    │            │    │                        │ │
│  │             │    │  chatbot   │    │  START → chatbot       │ │
│  │  messages:  │    │  tools     │    │  chatbot → tools_cond  │ │
│  │  list with  │    │  (ToolNode)│    │  tools → chatbot       │ │
│  │  add_msgs   │    │            │    │  chatbot → END         │ │
│  └────────────┘    └────────────┘    └────────────────────────┘ │
│                                                                  │
│  ┌────────────────────┐    ┌────────────────────────────────┐   │
│  │   Prebuilt         │    │   Checkpointing               │   │
│  │                    │    │                                │   │
│  │  ToolNode          │    │  MongoDBSaver                  │   │
│  │  tools_condition   │    │  thread_id: "7"               │   │
│  │                    │    │  Persists messages in MongoDB  │   │
│  └────────────────────┘    └────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.1 State — The Data That Flows Through the Graph

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

- **`TypedDict`**: Defines the shape of data flowing through the graph
- **`messages`**: A list of all conversation messages (HumanMessage, AIMessage, ToolMessage)
- **`Annotated[list, add_messages]`**: The `add_messages` reducer means returning `{"messages": [new]}` **appends** rather than replaces

### 7.2 Nodes — Functions That Process State

| Node | Type | What It Does |
|------|------|-------------|
| `chatbot` | Custom function | Calls GPT-4.1 with system prompt + messages, returns AI response |
| `tools` | `ToolNode` (prebuilt) | Executes tool calls from the AI's response |

### 7.3 Edges — How Nodes Connect

| Edge Type | From | To | Condition |
|-----------|------|----|-----------|
| Normal | START | chatbot | Always |
| **Conditional** | chatbot | tools OR END | `tools_condition` (has tool_calls?) |
| Normal | tools | chatbot | Always (loop back for follow-up) |

### 7.4 The Agent Loop (ReAct Pattern)

This project implements a **ReAct (Reason + Act)** loop:

```
1. REASON:  chatbot node → LLM thinks about what to do
2. ACT:     tools node → Execute the decided action (shell command)
3. OBSERVE: chatbot node → LLM sees the result, decides next step
4. REPEAT:  Until LLM responds without tool calls → END
```

### 7.5 `tools_condition` — The Decision Maker

```python
# This is a prebuilt function from LangGraph
# Pseudocode of what tools_condition does:
def tools_condition(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:        # AI wants to use a tool
        return "tools"                 # Route to tools node
    else:
        return "__end__"               # No tool needed → END
```

### 7.6 Checkpointing — Memory Across Runs

```python
# Without checkpointer:
graph = graph_builder.compile()              # Forgets everything after each run

# With checkpointer:
graph = graph_builder.compile(checkpointer=MongoDBSaver(...))
# Now every graph.stream() call with the same thread_id continues the conversation
```

**What gets saved in MongoDB:**
- All messages (human, AI, tool) for each `thread_id`
- The full state at each step (allows rewinding/replaying)

---

## 8. MongoDB Checkpointing & Docker Compose

### docker-compose.yml Breakdown

```yaml
services:
  mongodb:                              # Service name
    image: mongo                        # Official MongoDB Docker image
    restart: always                     # Auto-restart if container crashes
    ports:
      - '27017:27017'                   # Expose MongoDB on default port
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin # Root username
      MONGO_INITDB_ROOT_PASSWORD: admin # Root password
    volumes:
      - mongodb_data_v2:/data/db        # Persist data across container restarts

volumes:
  mongodb_data_v2:                      # Named volume for data persistence
```

### How Checkpointing Works

```
┌────────────────┐     ┌────────────────┐     ┌─────────────────┐
│   LangGraph    │     │  MongoDBSaver  │     │    MongoDB      │
│   graph.stream │────▶│  (checkpoint)  │────▶│    Container    │
│                │     │                │     │    Port 27017   │
│  thread_id: 7  │     │  Save state    │     │                 │
│                │     │  after each    │     │  Collection:    │
│                │◀────│  node execution│◀────│  checkpoints    │
│  Load previous │     │                │     │                 │
│  messages      │     │  Restore state │     │  Volume:        │
│                │     │  on next run   │     │  mongodb_data_v2│
└────────────────┘     └────────────────┘     └─────────────────┘
```

### MongoDB Connection String

```
mongodb://admin:admin@localhost:27017
         ─────┬───── ─────────┬──────
           user:pass     host:port
```

### Starting MongoDB

```bash
# Navigate to Lang_Project/ directory
docker compose up -d
# MongoDB is now running on localhost:27017
```

---

## 9. OpenAI Text-to-Speech (TTS)

The project includes a `speak()` function that uses **OpenAI's TTS API** but is **not integrated into the main loop** — it's defined and commented out as a demo.

### TTS Function Explained

```python
async def speak(text: str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",          # Lightweight, fast TTS model
        voice="coral",                     # One of several voice presets
        input=text,                        # The text to speak
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",            # Raw audio (uncompressed)
    ) as response:
        await LocalAudioPlayer().play(response)  # Play through speakers
```

### OpenAI TTS Voice Options

| Voice | Description |
|-------|-------------|
| alloy | Neutral, balanced |
| echo | Warm, conversational |
| fable | Expressive, storytelling |
| onyx | Deep, authoritative |
| nova | Friendly, upbeat |
| shimmer | Soft, gentle |
| **coral** | Used in this project |

### How to Integrate TTS into the Main Loop

Currently the AI response is only printed. To make it speak:

```python
# Inside the while True loop, after pretty_print():
import asyncio
# Get the final AI text response
final_text = event["messages"][-1].content
if final_text:
    asyncio.run(speak(final_text))
```

---

## 10. requirements.txt Breakdown

```
audioop-lts==0.2.1           # Audio operations (low-level audio processing)
SpeechRecognition==3.14.2    # Speech-to-text library
standard-aifc==3.13.0        # AIFF audio file format support
standard-chunk==3.13.0       # Chunked audio reading
typing_extensions==4.13.2    # Backport of typing features
```

| Package | Why It's Needed |
|---------|----------------|
| `SpeechRecognition` | Core STT library used in main.py |
| `audioop-lts` | Audio operations needed by SpeechRecognition for processing audio data. The `-lts` suffix = Long Term Support backport (audioop was removed from Python 3.13 stdlib) |
| `standard-aifc` | AIFF audio format support (removed from Python 3.13 stdlib, backported) |
| `standard-chunk` | Chunked audio reading (removed from Python 3.13 stdlib, backported) |
| `typing_extensions` | Extra typing features for older Python versions |

> **Note:** Other core dependencies like `langgraph`, `langchain-openai`, `openai`, `python-dotenv` are not listed here — they're in the main project's requirements or installed separately.

---

## 11. How to Run This Project

### Step 1: Start MongoDB

```bash
cd Lang_Project
docker compose up -d
```

### Step 2: Install Dependencies

```bash
pip install SpeechRecognition[audio] langgraph langchain-openai python-dotenv
pip install langgraph-checkpoint-mongodb
pip install -r requirements.txt
```

### Step 3: Set Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 4: Run the Application

```bash
# From the Lang_Project parent directory
python -m Lang_Project.app.main

# Or from within Lang_Project/
cd Lang_Project
python -m app.main
```

### Step 5: Speak!

```
Say something!
> (speak into microphone) "List all files in the current directory"
Processing audio...
You Said: list all files in the current directory
AI: I'll run the `ls` command for you.
Tool: run_command(cmd="ls")
AI: Here are the files in the current directory: ...
Say something!
```

---

## 12. Potential Improvements

| Improvement | Description |
|-------------|-------------|
| **Integrate TTS** | Call `speak()` after each AI response for full voice conversation |
| **Dynamic thread_id** | Let user choose/create conversation threads instead of hardcoded `"7"` |
| **Error handling** | Catch `sr.UnknownValueError` (speech not understood) and `sr.RequestError` (API error) |
| **Use Whisper** | Replace `recognize_google()` with `recognize_whisper()` or `recognize_openai()` for better accuracy |
| **Sandbox commands** | Run commands in a Docker container instead of `os.system()` for security |
| **Wake word** | Add a wake word ("Hey assistant") so it doesn't try to process every sound |
| **Multiple tools** | Add file reading, web search, or code analysis tools |

---

## 13. Quick Reference Cheat Sheet

### SpeechRecognition Quick Reference

```python
import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as src:
    r.adjust_for_ambient_noise(src)     # Calibrate
    r.pause_threshold = 2               # Allow 2s pauses
    audio = r.listen(src)               # Record until silence
    text = r.recognize_google(audio)    # Transcribe (free, no key)
```

### LangGraph Agent Quick Reference

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define Tool
@tool
def my_tool(arg: str):
    """Description for LLM"""
    return result

# 3. Bind tool to LLM
llm_with_tools = llm.bind_tools([my_tool])

# 4. Create node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. Build graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", ToolNode([my_tool]))
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

# 6. Compile with checkpointer
compiled = graph.compile(checkpointer=my_checkpointer)
```

### MongoDB Checkpointing Quick Reference

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

with MongoDBSaver.from_conn_string("mongodb://admin:admin@localhost:27017") as cp:
    graph = graph_builder.compile(checkpointer=cp)
    config = {"configurable": {"thread_id": "my-thread"}}
    graph.stream({"messages": [...]}, config, stream_mode="values")
```

---

> **Summary:** This project is a voice-controlled coding assistant that combines SpeechRecognition (for hearing), LangGraph with GPT-4.1 (for thinking and deciding), os.system via a bound tool (for acting), and MongoDB checkpointing (for remembering). The agent follows a ReAct loop — it reasons about the user's spoken request, acts by executing shell commands, observes the results, and responds.