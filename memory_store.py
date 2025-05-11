import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
memory_file = "memory.json"
conversation_memory = []

if os.path.exists(memory_file):
    with open(memory_file, "r", encoding="utf-8") as f:
        conversation_memory = json.load(f)
        for text in conversation_memory:
            vec = model.encode([text])[0].astype('float32')
            index.add(np.array([vec]))

def get_embedding(text):
    return model.encode([text])[0]

def save_to_memory(text):
    vec = get_embedding(text).astype('float32')
    index.add(np.array([vec]))
    conversation_memory.append(text)

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(conversation_memory, f, ensure_ascii=False, indent=2)

def fetch_relevant_memory(query, top_k=3):
    if index.ntotal == 0:
        return []
    vec = get_embedding(query).astype('float32')
    D, I = index.search(np.array([vec]), top_k)
    return [conversation_memory[i] for i in I[0] if i < len(conversation_memory)]

