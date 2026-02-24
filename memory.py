import json
import os

MEMORY_FILE = "conversation_memory.json"

def load_memory():
    # 1. Check if file exists
    if not os.path.exists(MEMORY_FILE):
        return {}

    # 2. Check if file is empty (size is 0 bytes)
    if os.path.getsize(MEMORY_FILE) == 0:
        return {}

    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        # 3. Handle cases where file has content but it's not valid JSON
        return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def get_user_history(phone):
    memory = load_memory()
    return memory.get(phone, [])

def append_user_history(phone, message):
    memory = load_memory()
    memory.setdefault(phone, []).append(message)
    save_memory(memory)
