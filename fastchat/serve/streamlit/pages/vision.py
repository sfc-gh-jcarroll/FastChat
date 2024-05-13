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
    title="Vision Direct Chat",
    icon="👀",
    wide_mode=True,
)


DEFAULT_MESSAGE = "Hello there! Let's chat?"

# Store conversation state in streamlit session
if "vision_conversation" not in st.session_state:
    st.session_state["vision_conversation"] = ConversationUI()
    st.session_state["vision_conversation"].add_message(
        ConversationMessage(role="assistant", content=DEFAULT_MESSAGE),
        render=False
    )
conversation_ui: ConversationUI = st.session_state["vision_conversation"]


# Sidebar

with sidebar_container:

    models = [
        "llava-v1.6-vicuna-13b",
        "llava-v1.6-vicuna-7b",
    ]
    selected_model_name = st.selectbox(
        "Choose a model to chat with:", models,
        help="[LLaVA](https://github.com/haotian-liu/LLaVA): an open large language and vision assistant")
    
    # Override model so it functions (TODO: Fix to use real models)
    selected_model_name = "snowflake/snowflake-arctic-instruct"

    temperature, top_p, max_new_tokens = get_parameters(st.container())


files, chat = st.columns([0.3, 0.7])


# Files area

""

with files:
    st.warning("This is a mockup app, the model can't yet see the image")
    img = st.file_uploader("Add an image:", type=["png", "jpeg", "jpg"])
    if img:
        st.image(img, use_column_width="auto")
    
    # TODO: Add examples

# Chat area

""

# Render the chat
conversation_ui.render_all(container=chat)
user_input = st.chat_input("Enter your message here.") or st.session_state.pop("regenerate", None)    

if user_input:
    conversation_ui.add_message(
        ConversationMessage(role="user", content=user_input),
        container=chat
    )

    chat_response(
        conversation_ui,
        selected_model_name,
        temperature,
        top_p,
        max_new_tokens,
        container=chat,
    )

response_controls = chat.container()

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
