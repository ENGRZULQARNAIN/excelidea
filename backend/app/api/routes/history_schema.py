from pydantic import BaseModel
from typing import List, Union, Dict, Any


class Message(BaseModel):
    """Schema for a single message in the conversation history."""
    role: str
    content: Union[str, Dict[str, Any]]


class History(BaseModel):
    """Schema for the conversation history."""
    history: List[Message]


# Example usage:
# history_data = {
#     "history": [
#         {"role": "human", "content": "hi"},
#         {"role": "ai", "content": {"pb_id": "1234567890", "content": "how can i assist you"}}
#     ]
# }
# history = History(**history_data) 