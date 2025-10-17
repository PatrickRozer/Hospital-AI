"""
api_server.py
FastAPI server for multilingual chatbot
"""
""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
from fastapi import FastAPI, Body
from pydantic import BaseModel
from health_translator.chatbot_translator import chatbot_reply
from health_translator.monitor import log_interaction

app = FastAPI(title="HealthAI Multilingual Chatbot")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    result = chatbot_reply(req.message)
    log_interaction(req.message, result)
    return {"reply": result["reply"], "sentiment": result["sentiment"], "lang": result["input_lang"]}

@app.get("/")
def root():
    return {"message": "HealthAI Chatbot API running."}
