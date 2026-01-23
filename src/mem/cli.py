import logging
import time
import sys
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from mem0 import Memory
import dspy
import litellm


from lib.custom_lm.lms import Lm_Glm, Lm_Ollama_Qwen3_4b
from mem.memory_tools import MemoryReActAgent

# Embedder/LLM endpoints and a dedicated persist dir per embedder
PERSIST_DIR = Path.home() / ".mem0_qwen3_4b"
lm = Lm_Glm
litellm.drop_params = True
logging.basicConfig(level=logging.DEBUG)


config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "zai/glm-4.5-flash",
            "temperature": 0.1,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "qwen3-embedding:4b",
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
            # Optional: ChromaDB Cloud configuration
            # "api_key": "your-chroma-cloud-api-key",
            # "tenant": "your-chroma-cloud-tenant-id",
        },
    },
}


def run_memory_agent_demo():
    """Demonstration of memory-enhanced ReAct agent."""
    load_dotenv()
    logging.info(os.environ["ZAI_API_KEY"])
    lm = Lm_Glm
    litellm.drop_params = True

    # Configure DSPy
    dspy.configure(lm=lm)

    # Optional: hard reset the on-disk Mem0 store to avoid dim-mismatch
    if os.getenv("MEM_CLEAR_DISK") == "1" or "--clear-disk" in sys.argv:
        if PERSIST_DIR.exists():
            print(f"⚠️  Deleting on-disk Mem0 store at {PERSIST_DIR}...")
            shutil.rmtree(PERSIST_DIR)
        else:
            print(f"ℹ️  No on-disk Mem0 store found at {PERSIST_DIR}.")

    # Initialize memory system
    memory = Memory.from_config(config)

    # Create our agent
    agent = MemoryReActAgent(memory)

    # Optional: clear all memories if requested
    if os.getenv("MEM_CLEAR") == "1" or "--clear" in sys.argv:
        print("⚠️  Clearing all memories for user 'default_user'...")
        print(agent.memory.clear_all_memories(user_id="default_user"))
        return

    # Sample conversation demonstrating memory capabilities
    print("🧠 Memory-Enhanced ReAct Agent Demo")
    print("=" * 50)

    conversations = [
        "Hi, I'm Alice and I love Italian food, especially pasta carbonara.",
        "I'm Alice. I prefer to exercise in the morning around 7 AM.",
        "I'm Alice. What do you remember about my food preferences?",
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n📝 User: {user_input}")

        try:
            response = agent(user_input=user_input, user_id="Alice")
            print(f"🤖 Agent: {response.response}")

        except Exception as e:
            print(f"❌ Error: {e}")


# Run the demonstration
if __name__ == "__main__":
    run_memory_agent_demo()
