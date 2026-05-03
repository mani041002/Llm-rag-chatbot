import os
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# Load embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- PDF FUNCTIONS --------
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def create_vector_store(chunks):
    if not chunks:
        return None, []

    embeddings = embed_model.encode(chunks)

    # Fix if single chunk
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

def retrieve_context(query, index, chunks):
    query_vec = embed_model.encode([query])

    if len(query_vec.shape) == 1:
        query_vec = query_vec.reshape(1, -1)

    D, I = index.search(np.array(query_vec), k=3)

    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return "\n".join(results)

# -------- LLM FUNCTION --------
def ask_llm(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        return "Error: " + response.text

    result = response.json()

    return result["choices"][0]["message"]["content"]