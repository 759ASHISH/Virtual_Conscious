from llama_ollama import query_llama
from memory_store import save_to_memory, fetch_relevant_memory

def chat_with_memory(user_input):
    relevant_memories = fetch_relevant_memory(user_input)
    memory_context = "\n".join(relevant_memories)
    prompt = f"Context:\n{memory_context}\n\nUser: {user_input}\nAI:"
    response = query_llama(prompt)
    save_to_memory(user_input)
    return response