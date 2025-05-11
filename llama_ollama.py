import subprocess

def query_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3.2", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()
