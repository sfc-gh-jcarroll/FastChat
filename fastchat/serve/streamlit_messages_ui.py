import streamlit as st
from schemas import Conversation, ConversationMessage


class ConversationUI:
    def __init__(self):
        self.reset()

    def reset(self):
        self.conversation: Conversation = Conversation()

    def add_message(self, message: ConversationMessage, container=None, render=True):
        self.conversation.add_message(message)
        if render:
            self.render_message(message, container)

    def render_all(self, container=None):
        for message in self.conversation.messages:
            self.render_message(message, container)

    def render_message(self, message: ConversationMessage, container=None):
        escaped_content = message.content.replace("$", "\$")
        if container is not None:
            container.chat_message(message.role).write(escaped_content)
        else:
            st.chat_message(message.role).write(escaped_content)

    def create_new_prompt(self):
        output = ""
        for index, message in enumerate(self.conversation.messages):
            output += f" {message.role.upper()}: {message.content}"
            if index > 1 and message.role == "assistant":
                output += f"</s>"

        return f"{output} ASSISTANT:"
