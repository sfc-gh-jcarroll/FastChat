from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from util import extract_diff


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Conversation(BaseModel):
    messages: List[ConversationMessage] = []

    streaming_msg: str = ""

    def reset_streaming(self):
        self.streaming_msg = ""

    def add_message(self, message: ConversationMessage):
        """
        Adds a message to the conversation
        """
        self.messages.append(message)

    def update_streaming_msg(self, new_msg: str):
        self.streaming_msg = new_msg

    def get_new_streaming_chuck(self, current_chuck: str) -> str:
        return extract_diff(self.streaming_msg, current_chuck)

    def reset_messages(self):
        self.messages = []
