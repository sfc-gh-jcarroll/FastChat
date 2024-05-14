### This is the main entrypoint for running the streamlit app version
### Typically launched with: `streamlit run fastchat/serve/streamlit/app.py`
###
### This can also be run as a standalone version calling models on replicate
### as follows:
###
### ```sh
### pip install -r fastchat/serve/streamlit/requirements.txt
### export USE_REPLICATE=true
### export REPLICATE_API_TOKEN="r8_ABC123..."
### streamlit run fastchat/serve/streamlit/app.py
### ```

import streamlit as st
from streamlit_feedback import streamlit_feedback

from messages_ui import ConversationUI
from schemas import ConversationMessage
from common import (
    get_models,
    get_parameters,
    chat_response,
    MODELS_HELP_STR,
    page_setup
)

sidebar_container = page_setup(
    title="Direct Chat with Open LLMs",
    icon="💬",
)

DEFAULT_MESSAGE = "Hello there! Let's chat?"

# Store conversation state in streamlit session
if "conversation_ui" not in st.session_state:
    st.session_state["conversation_ui"] = ConversationUI()
    st.session_state["conversation_ui"].add_message(
        ConversationMessage(role="assistant", content=DEFAULT_MESSAGE),
        render=False
    )
conversation_ui: ConversationUI = st.session_state["conversation_ui"]


# Sidebar

with sidebar_container:

    models = get_models()
    selected_model_name = st.selectbox(
        "Choose a model to chat with:", models,
        help=MODELS_HELP_STR)

    temperature, top_p, max_new_tokens = get_parameters(st.container())


# Main area

""

# Render the chat
conversation_ui.render_all()
user_input = st.chat_input("Enter your message here.") or st.session_state.pop("regenerate", None)

if user_input:
    conversation_ui.add_message(
        ConversationMessage(role="user", content=user_input)
    )

    chat_response(
        conversation_ui,
        selected_model_name,
        temperature,
        top_p,
        max_new_tokens,
    )

response_controls = st.container()

def clear_history():
    conversation_ui.conversation.reset_messages()
    conversation_ui.add_message(
        ConversationMessage(role="assistant", content=DEFAULT_MESSAGE),
        render=False
    )

def flag():
    st.toast("Flagged conversation as inappropriate", icon=":material/flag:")

def regenerate():
    st.session_state.regenerate = conversation_ui.conversation.messages[-2].content
    del conversation_ui.conversation.messages[-2:]

if len(conversation_ui.conversation.messages) > 1:
    # TODO: Big loading skeleton always briefly shows on the hosted app
    cols = response_controls.columns(4)

    cols[0].button("⚠️ &nbsp; Flag", use_container_width=True, on_click=flag)
    cols[1].button("🔄&nbsp; Regenerate", use_container_width=True, on_click=regenerate)
    cols[2].button(
        "🗑&nbsp; Clear history",
        use_container_width=True,
        on_click=clear_history,
    )

    with cols[3]:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            #optional_text_label="[Optional] Please provide an explanation",
            align="center",
            key=f"feedback_{len(conversation_ui.conversation.messages)}",
        )

        if feedback:
            st.toast("Feedback submitted!", icon=":material/rate_review:")
