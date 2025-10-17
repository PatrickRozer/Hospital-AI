import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def preprocess_text(text: str) -> str:
    """
    Simple text preprocessing that mimics model training.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Paths
MODEL_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot"
INDEX_FILE = os.path.join(MODEL_DIR, "faq_index.faiss")
FAQ_FILE = os.path.join(MODEL_DIR, "faq_with_answers.csv")
QUESTIONS_NPY = os.path.join(MODEL_DIR, "faq_questions.npy")

# Load vector index + LLM
retriever = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(INDEX_FILE)
faq_df = pd.read_csv(FAQ_FILE)
faq_questions = np.load(QUESTIONS_NPY, allow_pickle=True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def retrieve_context(query, k=3):
    q_emb = retriever.encode([query])
    D, I = index.search(q_emb, k)
    contexts = [faq_df.iloc[i]["Answer"] for i in I[0]]
    return "\n".join(contexts)

def generate_response(query):
    context = retrieve_context(query)
    input_text = f"Context:\n{context}\n\nUser: {query}\nAI:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

if __name__ == "__main__":
    print("🤖 Healthcare Chatbot ready! Type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = generate_response(user_query)
        print("Bot:", answer)
