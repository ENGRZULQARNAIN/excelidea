from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    """Schema for a single message in the conversation history."""
    role: str
    content: str


class History(BaseModel):
    """Schema for the conversation history."""
    history: List[Message]


# Example usage:
# history_data = {
#     "history": [
#         {"role": "human", "content": "hi"},
#         {"role": "ai", "content": "how can i assist you"}
#     ]
# }
# history = History(**history_data) 