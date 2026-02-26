# nexus/app/memory/session_store.py

from typing import List, Dict

from app.schemas.chat import Message



# 👇 No Redis imports needed!

class SessionStore:

    def __init__(self, window_size: int = 10, ttl: int = 3600):

        # Simple dictionary to hold history in RAM

        # Structure: { "session_id": [ {role, content}, ... ] }

        self._data: Dict[str, List[dict]] = {}

        self.window_size = window_size



    async def add_message(self, session_id: str, message: Message):

        """Save message to local memory."""

        if session_id not in self._data:

            self._data[session_id] = []

        

        # Add new message

        self._data[session_id].append(message.model_dump())



        # Sliding Window: Keep only the last N messages

        if len(self._data[session_id]) > self.window_size:

            self._data[session_id] = self._data[session_id][-self.window_size:]



    async def get_history(self, session_id: str) -> List[dict]:

        """Retrieve history from local memory."""

        return self._data.get(session_id, [])