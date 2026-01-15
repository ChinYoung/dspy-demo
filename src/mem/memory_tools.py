import datetime
import logging
import dspy
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """Store information in memory."""
        try:
            self.memory.add(content, user_id=user_id)
            msg = f"Stored memory: {content}"
            logging.info(msg)
            return msg
        except Exception as e:
            err = f"Error storing memory: {str(e)}"
            logging.error(err)
            return err

    def search_memories(
        self, query: str, user_id: str = "default_user", limit: int = 5
    ) -> str:
        """Search for relevant memories."""
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            if not results:
                msg = "No relevant memories found."
                logging.info(msg)
                return msg

            memory_text = "Relevant memories found:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            logging.info(memory_text)
            return memory_text
        except Exception as e:
            err = f"Error searching memories: {str(e)}"
            logging.error(err)
            return err

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Get all memories for a user."""
        try:

            results = self.memory.get_all(user_id=user_id)
            if not results:
                msg = "No memories found for this user."
                logging.info(msg)
                return msg

            memory_text = "All memories for user:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            logging.info(memory_text)
            return memory_text
        except Exception as e:
            err = f"Error retrieving memories: {str(e)}"
            logging.error(err)
            return err

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            self.memory.update(memory_id, new_content)
            msg = f"Updated memory with new content: {new_content}"
            logging.info(msg)
            return msg
        except Exception as e:
            err = f"Error updating memory: {str(e)}"
            logging.error(err)
            return err

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            msg = "Memory deleted successfully."
            logging.info(msg)
            return msg
        except Exception as e:
            err = f"Error deleting memory: {str(e)}"
            logging.error(err)
            return err

    def clear_all_memories(self, user_id: str = "default_user") -> str:
        """Delete all memories for a user by iterating and removing them."""
        try:
            results = self.memory.get_all(user_id=user_id)
            total = 0
            if results and isinstance(results, dict) and "results" in results:
                for item in results["results"]:
                    mem_id = (
                        item.get("id")
                        or item.get("memory_id")
                        or item.get("_id")
                        or item.get("uuid")
                    )
                    if mem_id:
                        self.memory.delete(mem_id)
                        total += 1
            msg = f"Cleared {total} memories for user {user_id}."
            logging.warning(msg)
            return msg
        except Exception as e:
            err = f"Error clearing memories: {str(e)}"
            logging.error(err)
            return err


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryQA(dspy.Signature):
    """
    You're an assistant and have access to memory tools.
    Whenever you answer a user's input, store the information to memory for future reference.
    """

    user_input: str = dspy.InputField(
        desc="User's input to the agent, store relevant info to memory."
    )
    user_id: str = dspy.InputField(default="default_user")
    response: str = dspy.OutputField()


class MemoryReActAgent(dspy.Module):
    """A ReAct agent enhanced with Mem0 memory capabilities."""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory = MemoryTools(memory)

        # Create tools list for ReAct
        self.tools = [
            self.memory.store_memory,
            self.memory.search_memories,
            self.memory.get_all_memories,
            get_current_time,
            self.set_reminder,
            self.get_preferences,
            self.update_preferences,
        ]

        # Initialize ReAct with our tools
        self.react = dspy.ReAct(signature=MemoryQA, tools=self.tools, max_iters=6)

    def forward(self, user_input: str, user_id: str = "default_user"):
        """Process user input with memory-aware reasoning and persist the turn."""
        # Run ReAct to get a response
        result = self.react(user_input=user_input, user_id=user_id)
        logging.info(f"Agent response: {result.response}")
        return result

    def set_reminder(
        self,
        reminder_text: str,
        date_time: Optional[str] = None,
        user_id: str = "default_user",
    ) -> str:
        """Set a reminder for the user."""
        reminder = f"Reminder set for {date_time}: {reminder_text}"
        msg = self.memory.store_memory(f"REMINDER: {reminder}", user_id=user_id)
        logging.info(msg)
        return msg

    def get_preferences(
        self, category: str = "general", user_id: str = "default_user"
    ) -> str:
        """Get user preferences for a specific category."""
        query = f"user preferences {category}"
        logging.info(f"Searching preferences with query: {query}")
        msg = self.memory.search_memories(query=query, user_id=user_id)
        logging.info(msg)
        return msg

    def update_preferences(
        self, category: str, preference: str, user_id: str = "default_user"
    ) -> str:
        """Update user preferences."""
        preference_text = f"User preference for {category}: {preference}"
        logging.info(f"Updating preferences with text: {preference_text}")
        msg = self.memory.store_memory(preference_text, user_id=user_id)
        logging.info(msg)
        return msg
