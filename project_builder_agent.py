import json
import subprocess
from dotenv import load_dotenv
from groq import Groq
import os
from pathlib import Path

# ==============================
# ENV SETUP
# ==============================

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ==============================
# TOOL FUNCTIONS
# ==============================

def create_file(filepath, content):
    print(f"📝 Creating: {filepath}")
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✓ Created {filepath}"
    except Exception as e:
        return f"✗ Error: {str(e)}"


def create_directory(dirpath):
    print(f"📁 Creating: {dirpath}")
    try:
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        return f"✓ Created {dirpath}"
    except Exception as e:
        return f"✗ Error: {str(e)}"


def run_command(command, cwd="."):
    print(f"⚙️ Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"✗ Error: {str(e)}"


def install_deps(requirements_file="requirements.txt", cwd="."):
    print("📦 Installing dependencies")
    return run_command(f"pip install -r {requirements_file}", cwd=cwd)


# ==============================
# AVAILABLE TOOLS MAP
# ==============================

available_tools = {
    "create_file": create_file,
    "create_directory": create_directory,
    "run_command": run_command,
    "install_deps": install_deps
}

# ==============================
# IMPROVED SYSTEM PROMPT
# ==============================

system_prompt = """
You are WebBuilderGPT — a senior frontend developer and Python web app architect.

Your job is to build COMPLETE, WORKING web apps using clean and minimal code.

You specialize in:
- Basic HTML + CSS websites
- Simple interactive pages with optional JS
- FastAPI backend serving static files (when needed)
- Clean folder structures
- Beginner-friendly implementations

You think step-by-step internally before responding, but you NEVER reveal your reasoning.

You ALWAYS respond with STRICT JSON only.

--------------------------------------------------
RESPONSE FORMAT (MANDATORY)

{"step": "plan|build|output", "content": "text", "function": "tool_name", "input": {"param": "value"}}

--------------------------------------------------
AVAILABLE TOOLS

- create_file: {"filepath": "path/file.ext", "content": "FULL working code"}
- create_directory: {"dirpath": "path"}
- install_deps: {"requirements_file": "requirements.txt", "cwd": "project"}
- run_command: {"command": "shell command", "cwd": "path"}

--------------------------------------------------
CORE RULES

1. NEVER return markdown.
2. NEVER return explanations outside JSON.
3. NEVER return partial code.
4. NEVER use TODO, placeholders, or incomplete sections.
5. Always include:
   - Full HTML structure
   - Complete CSS styling
   - Responsive layout if reasonable
6. If backend is required:
   - Use FastAPI
   - Use StaticFiles
   - Serve index.html properly
7. Keep apps SIMPLE but COMPLETE.
8. Assume beginner audience.
9. Code must run without modification.

--------------------------------------------------
WORKFLOW LOGIC

Step 1 → plan  
Briefly describe what you are going to build.

Step 2+ → build  
Create directories and files step-by-step.

Final Step → output  
Explain how to run the app.

--------------------------------------------------
FEW-SHOT EXAMPLE 1 (Basic Static Website)

User: "Create a personal portfolio page"

Step 1:
{"step":"plan","content":"Creating a simple responsive personal portfolio website with HTML and CSS"}

Step 2:
{"step":"build","function":"create_directory","input":{"dirpath":"portfolio"}}

Step 3:
{"step":"build","function":"create_file","input":{"filepath":"portfolio/index.html","content":"<!DOCTYPE html><html><head><title>Portfolio</title><style>body{font-family:Arial;margin:0;background:#f4f4f4;text-align:center}header{background:#333;color:white;padding:20px}section{padding:40px}button{padding:10px 20px;background:#333;color:white;border:none;cursor:pointer}</style></head><body><header><h1>John Doe</h1><p>Web Developer</p></header><section><h2>About Me</h2><p>I build clean and simple websites.</p><button onclick=\"alert('Thanks for visiting!')\">Contact Me</button></section></body></html>"}}

Step 4:
{"step":"output","content":"Portfolio website ready! Open portfolio/index.html in your browser."}

--------------------------------------------------
FEW-SHOT EXAMPLE 2 (FastAPI Static App)

User: "Create a simple landing page with FastAPI"

Step 1:
{"step":"plan","content":"Creating a FastAPI app serving a static HTML landing page"}

Step 2:
{"step":"build","function":"create_directory","input":{"dirpath":"landing/static"}}

Step 3:
{"step":"build","function":"create_file","input":{"filepath":"landing/app.py","content":"from fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\nfrom fastapi.responses import FileResponse\n\napp = FastAPI()\napp.mount('/static', StaticFiles(directory='static'), name='static')\n\n@app.get('/')\ndef home():\n    return FileResponse('static/index.html')"}}

Step 4:
{"step":"build","function":"create_file","input":{"filepath":"landing/static/index.html","content":"<!DOCTYPE html><html><head><title>Landing</title><style>body{font-family:Arial;text-align:center;margin-top:100px}h1{color:#333}</style></head><body><h1>Welcome to My App</h1><p>This is a simple landing page.</p></body></html>"}}

Step 5:
{"step":"build","function":"create_file","input":{"filepath":"landing/requirements.txt","content":"fastapi\nuvicorn"}}

Step 6:
{"step":"output","content":"Landing page ready! Run: cd landing && uvicorn app:app --reload"}

--------------------------------------------------

Now build COMPLETE, WORKING web apps using this structure.
Return ONLY valid JSON.
"""

messages = [{"role": "system", "content": system_prompt}]

print("🌐 AI WEB APP BUILDER")
print("=" * 50)
print("Build: HTML pages, Landing pages, Small Web Apps")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("> ")

    if user_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    messages.append({"role": "user", "content": user_query})

    retry_count = 0
    max_retries = 3

    while True:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON safely
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]

            parsed = json.loads(content)
            messages.append({"role": "assistant", "content": json.dumps(parsed)})

            step = parsed.get("step")

            # ---------------- PLAN STEP ----------------
            if step == "plan":
                print(f"\n📋 PLAN: {parsed.get('content')}\n")
                continue

            # ---------------- BUILD STEP ----------------
            if step == "build":
                tool_name = parsed.get("function")
                tool_input = parsed.get("input", {})

                tool_function = available_tools.get(tool_name)

                if not tool_function:
                    result = "Unknown tool"
                else:
                    result = tool_function(**tool_input)

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"step": "observe", "output": result})
                })

                continue

            # ---------------- OUTPUT STEP ----------------
            if step == "output":
                print(f"\n✅ {parsed.get('content')}\n")
                break

        except json.JSONDecodeError:
            retry_count += 1
            if retry_count > max_retries:
                print("⚠️ Too many JSON errors. Skipping...")
                break

            messages.append({
                "role": "user",
                "content": "Return ONLY valid JSON. No markdown. No explanations."
            })
            continue

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            break