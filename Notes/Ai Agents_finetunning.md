# AI Agents, LLMs & Fine-Tuning — Revision Notes

---

## Part 1: LLMs (Large Language Models)

### What is an LLM?

An LLM is a large neural network (based on the **Transformer architecture**) trained on massive amounts of text data. It learns patterns in language and can **predict the next token** given a sequence of tokens.

> Think of it as: A super-smart autocomplete that has read the entire internet.

### How does an LLM work (simplified)?

1. **Tokenization** — Text is broken into tokens (words/sub-words/characters)
2. **Vector Embedding** — Each token is converted into a numerical vector
3. **Self-Attention** — The model figures out which tokens are related to each other
4. **Feed Forward Network** — Processes the attention output
5. **Output Generation** — Predicts the next most probable token

### What gives LLMs their power?

- **Scale** — Trained on billions of parameters and terabytes of data
- **Transformer Architecture** — Self-attention mechanism lets them understand context across long sequences
- **Pre-training + Fine-tuning** — First learn general language, then specialize for tasks
- **In-context Learning** — Can follow instructions and examples given in the prompt itself (zero-shot, few-shot)

---

## Part 2: From Simple Chat to Powerful Agents — The Journey

### Step 1: Basic LLM API Call (`chat.py`)

The simplest way to use an LLM — send a message, get a response.

```python
from openai import OpenAI

client = OpenAI()

result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        { "role": "user", "content": "What is greator? 9.8 or 9.11" }
    ]
)
print(result.choices[0].message.content)
```

**Key Concept:** The `messages` array is the conversation history. Each message has a `role` (`system`, `user`, `assistant`).

---

### Step 2: System Prompt — Giving the LLM a Personality (`chat_2.py`)

By adding a **system prompt**, we control HOW the LLM behaves.

```python
system_prompt = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.
"""

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is a mobile phone?" }
    ]
)
```

**Key Concept:** The system prompt is like giving the LLM a job description. It follows these instructions for the entire conversation.

---

### Step 3: Chain-of-Thought with JSON Output (`chat_3.py` & `chat_3_auto.py`)

Instead of one-shot answers, we make the LLM **think step by step** using structured JSON output.

```python
# Force JSON output format
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=messages
)
```

**The Agentic Loop (chat_3_auto.py):**
```
User Input → LLM Analyses → LLM Thinks → LLM Thinks Again → Output → Validate → Final Result
```

The steps follow a sequence: `analyse` → `think` → `output` → `validate` → `result`

Each step is one LLM call. The output of each step is appended to `messages` and fed back to the LLM for the next step. This is the **agentic loop** — the LLM keeps calling itself in a `while True` loop until it reaches the final result.

```python
while True:
    response = client.chat.completions.create(...)
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({ "role": "assistant", "content": json.dumps(parsed_response) })

    if parsed_response.get("step") != "output":
        print(f"🧠: {parsed_response.get('content')}")
        continue

    print(f"🤖: {parsed_response.get('content')}")
    break
```

---

### Step 4: Different LLM Providers

**Gemini (Google) — `chat_gemini.py`:**
```python
from google import genai

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
response = client.models.generate_content(
    model='gemini-1.5-flash', contents='Why is the sky blue?'
)
```

**Groq (fast inference) — used in `weather_agent.py`:**
```python
from groq import Groq

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    response_format={"type": "json_object"},
    messages=messages
)
```

**Ollama (self-hosted via Docker) — `ollama_api.py`:**
```python
from ollama import Client

client = Client(host="http://localhost:11434")
client.pull('qwen2.5:0.5b')  # Pull the model

response = client.chat(model="qwen2.5:0.5b", messages=[
    {"role": "user", "content": message}
])
```

Docker Compose to run Ollama:
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - '11434:11434'
    volumes:
      - models:/root/.ollama/models
```

### Deep Dive: Ollama API (`ollama_api.py`) — Self-Hosted LLM as a REST API

This file wraps a **self-hosted Ollama model** behind a **FastAPI** server, so you can call your own LLM via HTTP — no OpenAI/Groq/Gemini API key needed.

#### Full Code:

```python
from fastapi import FastAPI, Body
from ollama import Client

app = FastAPI()

client = Client(
    host="http://localhost:11434"
)

# Pull model on startup
client.pull('qwen2.5:0.5b')

@app.post("/chat")
def chat(message: str = Body(..., embed=True)):
    response = client.chat(model="qwen2.5:0.5b", messages=[
        {"role": "user", "content": message}
    ])
    return {"message": response['message']['content']}
```

#### How it works — Step by Step:

| Step | What happens |
|------|-------------|
| **1. Docker Compose Up** | Runs the Ollama server container on port `11434` |
| **2. `Client(host=...)`** | Python connects to the Ollama server running in Docker |
| **3. `client.pull('qwen2.5:0.5b')`** | Downloads the `qwen2.5:0.5b` model (like `docker pull` but for AI models). Only runs once — cached after that |
| **4. FastAPI `@app.post("/chat")`** | Exposes a POST endpoint at `/chat` |
| **5. `Body(..., embed=True)`** | Expects JSON body like `{"message": "Hello"}`. The `embed=True` means the field is wrapped inside a JSON object |
| **6. `client.chat(...)`** | Sends the user message to the Ollama model and gets a response |
| **7. Returns JSON** | `{"message": "The model's response here"}` |

#### How to Run:

```bash
# 1. Start Ollama via Docker
docker compose up -d

# 2. Run the FastAPI server
uvicorn ollama_api:app --reload
```

Then call it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'
```

#### Key Concepts:

- **Ollama** — A tool to run open-source LLMs (LLaMA, Qwen, Mistral, etc.) locally on your machine
- **`qwen2.5:0.5b`** — A small 0.5 billion parameter model (lightweight, fast, good for testing)
- **FastAPI** — Python web framework for building APIs. Auto-generates Swagger docs at `/docs`
- **Why self-host?** — No API costs, full privacy (data never leaves your machine), works offline
- **Docker volume `models:`** — Persists downloaded models so they survive container restarts

#### Ollama vs Cloud APIs:

| | Ollama (Self-hosted) | OpenAI / Groq / Gemini |
|---|---|---|
| **Cost** | Free (your hardware) | Pay per token |
| **Privacy** | Data stays local | Data sent to cloud |
| **Speed** | Depends on your GPU/CPU | Fast (optimized infra) |
| **Model Quality** | Open-source models | Proprietary (GPT-4, Claude) |
| **Internet** | Works offline | Requires internet |
| **Setup** | Docker + pull model | Just an API key |

---

## Part 3: AI Agents — LLMs with Superpowers (Tools)

### What is an AI Agent?

An AI Agent = **LLM + Tools + Agentic Loop**

- **LLM** = The brain (understands and plans)
- **Tools** = The hands (can actually DO things — call APIs, run code, etc.)
- **Agentic Loop** = The cycle of Plan → Action → Observe → Repeat

> A plain LLM can only generate text. An Agent can **take actions in the real world**.

### The Weather Agent (`weather_agent.py`) — Full Breakdown

#### Tools Defined:

```python
def get_weather(city: str):
    """Calls wttr.in API to get weather for a city"""
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    return f"The weather in {city} is {response.text}."

def run_command(command):
    """Executes a system command (shell command) and returns output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr
```

#### Tool Registry:

```python
avaiable_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output"
    }
}
```

The LLM knows about these tools through the **system prompt**. It decides WHICH tool to call and WITH WHAT input.

#### The Agent Loop — Plan → Action → Observe → Output:

```
User: "What is the weather in Delhi?"

🧠 Step: plan    → "The user wants weather data for Delhi"
🧠 Step: plan    → "I should call get_weather tool"
🔨 Step: action  → Calls get_weather("Delhi")
👀 Step: observe → "The weather in Delhi is Sunny +32°C"
🤖 Step: output  → "The weather in Delhi is Sunny, 32°C"
```

In code:

```python
while True:  # Outer loop — keeps taking user input
    user_query = input('> ')
    messages.append({ "role": "user", "content": user_query })

    while True:  # Inner loop — the agentic loop
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            messages=messages
        )
        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant", "content": json.dumps(parsed_output) })

        if parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get('content')}")
            continue  # Keep planning

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")
            # Actually call the tool function!
            output = avaiable_tools[tool_name]["fn"](tool_input)
            # Feed observation back to the LLM
            messages.append({ "role": "assistant", "content": json.dumps({ "step": "observe", "output": output }) })
            continue  # Let LLM process the observation

        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get('content')}")
            break  # Done!
```

### How `sum.py` was Created via the Agent

The `run_command` tool gives the agent the power to **execute system commands**. So when you ask:

> "Create a Python file called sum.py that adds two numbers"

The agent:
1. **Plans** — "I need to create a file with Python code"
2. **Actions** — Calls `run_command` with something like:
   ```
   echo "def sum(a,b):\n    return a + b\nprint(sum(5,10))" > sum.py
   ```
3. **Observes** — File created successfully
4. **Outputs** — "I've created sum.py with a sum function"

The resulting file (`sum.py`):
```python
def sum(a,b):
    return a + b
print(sum(5,10))
```

> **This is the power of tools!** The LLM alone can't create files. But with `run_command` as a tool, it can execute any shell command — create files, install packages, run scripts, etc.

---

## Part 4: Tokenization & Embeddings — The Foundation

### Tokenization (`tokenization.py`)

Tokenization = Breaking text into tokens (smallest units the model understands).

```python
import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')
print("Vocab Size", encoder.n_vocab)  # Total tokens the model knows

text = "The cat sat on the mat"
tokens = encoder.encode(text)   # Text → Token IDs
print("Tokens", tokens)         # [976, 9059, 10139, 402, 290, 2450]

decoded = encoder.decode(tokens) # Token IDs → Text
print("Decoded", decoded)        # "The cat sat on the mat"
```

**Key Point:** Models don't see text — they see numbers (token IDs). Each model has its own tokenizer and vocab size.

### Vector Embeddings (`embedding.py`)

Embeddings = Converting tokens into **numerical vectors** that capture semantic meaning.

```python
response = client.embeddings.create(
    input="Eiffel Tower is in Paris and is a famous landmark",
    model="text-embedding-3-small"
)
print("Vector Embeddings", response.data[0].embedding)  # List of floats
```

**Key Point:** Similar meanings → vectors close together. "King" and "Queen" have similar embeddings. "King" and "Banana" don't.

---

## Part 5: Fine-Tuning LLMs (Google Colab + Hugging Face)

### What is Fine-Tuning?

Fine-tuning = Taking a **pre-trained LLM** and training it further on **your specific data** so it becomes better at your specific task.

> Pre-trained LLM is like a topper student. Fine-tuning is like giving them coaching for a specific exam (UPSC, JEE, etc.)

### Setup: Google Colab with GPU

- Open [Google Colab](https://colab.research.google.com/)
- Go to **Runtime → Change runtime type → GPU** (T4 GPU is free tier)
- GPU is needed because fine-tuning involves matrix operations that are massively parallelized on GPUs

### Step-by-Step: Loading & Using a Model from Hugging Face

#### 1. Install Dependencies

```python
!pip install transformers torch accelerate
```

#### 2. Import the Key Classes

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
```

| Class | What it does |
|-------|-------------|
| `AutoTokenizer` | Automatically loads the correct tokenizer for any model |
| `AutoModelForCausalLM` | Loads a causal language model (text generation model — predicts next token) |
| `pipeline` | High-level API that wraps tokenizer + model for easy use |

#### 3. Authenticate with Hugging Face

Some models (like LLaMA) are gated — you need a Hugging Face token.

```python
from huggingface_hub import login

login(token="hf_YOUR_TOKEN_HERE")
# Or set environment variable: HF_TOKEN
```

Get your token from: https://huggingface.co/settings/tokens

#### 4. Load Tokenizer & Model

```python
model_name = "meta-llama/Llama-2-7b-hf"  # or any model from HF

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

- `AutoTokenizer.from_pretrained(model_name)` — Downloads & loads the tokenizer for that model
- `AutoModelForCausalLM.from_pretrained(model_name)` — Downloads & loads the model weights

#### 5. Tokenize Input (Manual Way)

```python
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt")  # "pt" = PyTorch tensors

input_ids = inputs["input_ids"]  # Token IDs as tensor
print(input_ids)  # tensor([[  450, 3105,  310, 319, 29902,  338]])
```

- `input_ids` — The tokenized version of your text (numbers the model understands)
- `return_tensors="pt"` — Returns PyTorch tensors (needed for the model)

#### 6. Generate / Predict Text

```python
output = model.generate(input_ids, max_new_tokens=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

- `model.generate(input_ids)` — Takes token IDs, predicts next tokens
- `max_new_tokens=50` — Generate up to 50 new tokens
- `tokenizer.decode()` — Converts token IDs back to readable text
- `skip_special_tokens=True` — Removes special tokens like `<s>`, `</s>`, `<pad>`

#### 7. Easy Way: Using `pipeline`

```python
gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

result = gen_pipeline("The future of AI is", max_new_tokens=50)
print(result[0]["generated_text"])
```

- `pipeline("text-generation", ...)` — Creates a text generation pipeline
- Handles tokenization, model inference, and decoding all in one call
- Other pipeline tasks: `"sentiment-analysis"`, `"summarization"`, `"translation"`, etc.

### The Complete Flow Visualized

```
Your Text
   ↓
AutoTokenizer  →  input_ids (numbers)
   ↓
AutoModelForCausalLM  →  generates new token IDs
   ↓
tokenizer.decode()  →  Human readable text
```

Or simply:
```
pipeline("text-generation")  →  Does ALL of the above in one line
```

### Fine-Tuning Key Concepts

| Term | Meaning |
|------|---------|
| **Pre-trained Model** | Model already trained on huge data (general knowledge) |
| **Fine-tuning** | Further training on your specific dataset |
| **LoRA / QLoRA** | Efficient fine-tuning — only trains a small % of parameters |
| **Epochs** | Number of times the model sees your entire dataset |
| **Learning Rate** | How big the steps are during training (too high = chaos, too low = slow) |
| **Dataset** | Your custom data in format like `{"instruction": "...", "output": "..."}` |

---

## Quick Revision Cheat Sheet

| Concept | One-liner |
|---------|-----------|
| **LLM** | Neural network that predicts the next token |
| **Tokenization** | Text → Numbers (token IDs) |
| **Embeddings** | Token IDs → Vectors (semantic meaning) |
| **System Prompt** | Job description for the LLM |
| **Chain-of-Thought** | Force the LLM to think step by step |
| **JSON Mode** | `response_format={"type": "json_object"}` — structured output |
| **Agentic Loop** | Plan → Action → Observe → Repeat until done |
| **Tools** | Functions the LLM can call (weather API, shell commands, etc.) |
| **Agent** | LLM + Tools + Agentic Loop |
| **AutoTokenizer** | Loads the right tokenizer for any HF model |
| **AutoModelForCausalLM** | Loads a text-generation model from HF |
| **input_ids** | Tokenized text as tensor — input to the model |
| **model.generate()** | Predict/generate new tokens from input_ids |
| **pipeline** | High-level wrapper: tokenize + generate + decode in one call |
| **Fine-tuning** | Train a pre-trained model on your own data |
| **Google Colab GPU** | Free T4 GPU for training/inference |
| **HF Token** | Authentication token for gated models on Hugging Face |
