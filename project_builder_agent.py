import json
import subprocess
from dotenv import load_dotenv
from groq import Groq
import os
from pathlib import Path

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

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
    print(f"⚙️  Running: {command}")
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"✗ Error: {str(e)}"

def git_init(dirpath="."):
    print(f"🔧 Git init: {dirpath}")
    return run_command("git init", cwd=dirpath)

def git_add(files=".", cwd="."):
    print(f"➕ Git add: {files}")
    return run_command(f"git add {files}", cwd=cwd)

def git_commit(message, cwd="."):
    print(f"💾 Git commit: {message}")
    return run_command(f'git commit -m "{message}"', cwd=cwd)

def git_push(cwd="."):
    print(f"🚀 Git push")
    return run_command("git push", cwd=cwd)

def install_deps(requirements_file="requirements.txt", cwd="."):
    print(f"📦 Installing dependencies")
    return run_command(f"pip install -r {requirements_file}", cwd=cwd)

def run_server(command="uvicorn app:app --reload", cwd="."):
    print(f"🌐 Starting server: {command}")
    return f"Server command: {command}\nRun manually: cd {cwd} && {command}"

available_tools = {
    "create_file": {"fn": create_file},
    "create_directory": {"fn": create_directory},
    "run_command": {"fn": run_command},
    "git_init": {"fn": git_init},
    "git_add": {"fn": git_add},
    "git_commit": {"fn": git_commit},
    "git_push": {"fn": git_push},
    "install_deps": {"fn": install_deps},
    "run_server": {"fn": run_server}
}

system_prompt = """
You are a Python Web App Builder. Create COMPLETE, WORKING apps.

IMPORTANT: Return ONLY valid JSON. No markdown, no extra text.

RESPONSE FORMAT:
{"step": "plan|build|output", "content": "text", "function": "tool_name", "input": {"param": "value"}}

TOOLS:
- create_file: {"filepath": "path/file.py", "content": "COMPLETE code"}
- create_directory: {"dirpath": "path"}
- install_deps: {"requirements_file": "requirements.txt", "cwd": "project"}
- run_command: {"command": "any shell command", "cwd": "path"}

CRITICAL RULES:
1. Write COMPLETE, WORKING code - no placeholders, no TODO comments
2. Include ALL imports, ALL functions, ALL logic
3. For games: Include full game logic + HTML/CSS/JS frontend
4. For FastAPI: Use StaticFiles to serve HTML, include all routes
5. Keep code SIMPLE but COMPLETE
6. Test logic mentally before generating

EXAMPLE - Tic Tac Toe:
Step 1: {"step": "plan", "content": "Creating tic-tac-toe with FastAPI backend, game logic, and interactive HTML/JS frontend"}
Step 2: {"step": "build", "function": "create_directory", "input": {"dirpath": "tictactoe/static"}}
Step 3: {"step": "build", "function": "create_file", "input": {"filepath": "tictactoe/app.py", "content": "from fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\nfrom fastapi.responses import FileResponse\n\napp = FastAPI()\napp.mount('/static', StaticFiles(directory='static'), name='static')\n\n@app.get('/')\ndef home():\n    return FileResponse('static/index.html')"}}
Step 4: {"step": "build", "function": "create_file", "input": {"filepath": "tictactoe/static/index.html", "content": "<!DOCTYPE html>\n<html>\n<head><title>Tic Tac Toe</title><style>body{font-family:Arial;text-align:center}.cell{width:100px;height:100px;border:2px solid black;display:inline-block;font-size:48px;line-height:100px;cursor:pointer}</style></head>\n<body><h1>Tic Tac Toe</h1><div id='board'></div><p id='status'>Player X's turn</p><button onclick='reset()'>Reset</button><script>let board=['','','','','','','','',''];let currentPlayer='X';let gameActive=true;const winPatterns=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];function init(){const boardDiv=document.getElementById('board');boardDiv.innerHTML='';for(let i=0;i<9;i++){const cell=document.createElement('div');cell.className='cell';cell.onclick=()=>makeMove(i);cell.id='cell-'+i;boardDiv.appendChild(cell)}}function makeMove(index){if(board[index]||!gameActive)return;board[index]=currentPlayer;document.getElementById('cell-'+index).textContent=currentPlayer;if(checkWin()){document.getElementById('status').textContent='Player '+currentPlayer+' wins!';gameActive=false;return}if(board.every(cell=>cell)){document.getElementById('status').textContent='Draw!';gameActive=false;return}currentPlayer=currentPlayer==='X'?'O':'X';document.getElementById('status').textContent='Player '+currentPlayer+\\'s turn'}function checkWin(){return winPatterns.some(pattern=>pattern.every(index=>board[index]===currentPlayer))}function reset(){board=['','','','','','','','',''];currentPlayer='X';gameActive=true;init();document.getElementById('status').textContent='Player X\\'s turn'}init()</script></body></html>"}}
Step 5: {"step": "build", "function": "create_file", "input": {"filepath": "tictactoe/requirements.txt", "content": "fastapi\nuvicorn"}}
Step 6: {"step": "build", "function": "install_deps", "input": {"cwd": "tictactoe"}}
Step 7: {"step": "output", "content": "Tic-tac-toe game ready! Run: cd tictactoe && uvicorn app:app --reload\nThen visit: http://localhost:8000"}

Now build apps with COMPLETE, WORKING code!
"""

messages = [{"role": "system", "content": system_prompt}]

print("🎮 SIMPLE APP BUILDER")
print("=" * 50)
print("Build: Games, Calculators, Web Apps")
print("I create COMPLETE, WORKING code!\n")

while True:
    user_query = input('> ')
    if user_query.lower() in ['exit', 'quit']: break
    
    messages.append({"role": "user", "content": user_query})
    
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown and extract JSON
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        content = part
                        break
            
            # Find JSON object
            if "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                content = content[start:end]
            
            parsed = json.loads(content)
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            
            if parsed.get("step") == "plan":
                print(f"📋 {parsed.get('content')}")
                continue
            
            if parsed.get("step") == "build":
                tool = parsed.get("function")
                inp = parsed.get("input", {})
                
                if tool == "create_file":
                    result = create_file(inp.get("filepath"), inp.get("content", ""))
                elif tool == "create_directory":
                    result = create_directory(inp.get("dirpath"))
                elif tool == "run_command":
                    result = run_command(inp.get("command"), inp.get("cwd", "."))
                elif tool == "git_init":
                    result = git_init(inp.get("dirpath", "."))
                elif tool == "git_add":
                    result = git_add(inp.get("files", "."), inp.get("cwd", "."))
                elif tool == "git_commit":
                    result = git_commit(inp.get("message", "Update"), inp.get("cwd", "."))
                elif tool == "git_push":
                    result = git_push(inp.get("cwd", "."))
                elif tool == "install_deps":
                    result = install_deps(inp.get("requirements_file", "requirements.txt"), inp.get("cwd", "."))
                elif tool == "run_server":
                    result = run_server(inp.get("command", "uvicorn app:app --reload"), inp.get("cwd", "."))
                else:
                    result = "Unknown tool"
                
                messages.append({"role": "assistant", "content": json.dumps({"step": "observe", "output": result})})
                continue
            
            if parsed.get("step") == "output":
                print(f"\n✅ {parsed.get('content')}\n")
                break
                
        except json.JSONDecodeError:
            retry_count += 1
            if retry_count > max_retries:
                print("⚠️ Too many JSON errors, skipping...")
                break
            messages.append({"role": "user", "content": "Return ONLY valid JSON, no markdown"})
            continue
        except Exception as e:
            print(f"❌ {str(e)}")
            break
