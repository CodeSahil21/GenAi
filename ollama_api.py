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
