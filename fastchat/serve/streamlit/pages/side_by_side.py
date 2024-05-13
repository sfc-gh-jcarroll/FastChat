import threading
import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from messages_ui import ConversationUI
from schemas import ConversationMessage
from common import (
    chat_response,
    get_models,
    get_parameters,
    MODELS_HELP_STR,
    page_setup,
)

sidebar_container = page_setup(
        title="Benchmarking LLMs in the Wild",
        icon="⚔️",
        wide_mode=not st.session_state.get("share_mode"),
        collapse_sidebar=st.session_state.get("share_mode"),
    )

DEFAULT_MESSAGE = "Hello there! Let's chat?"

# Store conversation state in streamlit session
if "side_by_side" not in st.session_state:
    st.session_state["side_by_side"] = [ConversationUI(), ConversationUI()]
    for conversation in st.session_state["side_by_side"]:
        conversation.add_message(
            ConversationMessage(role="assistant", content=DEFAULT_MESSAGE),
            render=False
        )
conversations = st.session_state["side_by_side"]


# Sidebar
temperature, top_p, max_new_tokens = get_parameters(sidebar_container)


# Main area

""

models = get_models()
MODEL_LABELS = ["Model A", "Model B"]
selected_models = [None for _ in conversations]
model_cols = st.columns(len(selected_models))
for idx in range(len(selected_models)):
    selected_models[idx] = model_cols[idx].selectbox(
        f"Select {MODEL_LABELS[idx]}:", models,
        help=MODELS_HELP_STR, key=f"model_select_{idx}")

# Render the chat
for idx, msg in enumerate(conversations[0].conversation.messages):
    if msg.role == "user":
        conversations[0].render_message(msg)
    else:
        msg_cols = st.columns(len(conversations))
        for i, conv in enumerate(conversations):
            conv.render_message(
                conv.conversation.messages[idx],
                container=msg_cols[i],
            )

user_msg = st.empty()
response = st.empty()
feedback_controls = st.empty()
response_controls = st.empty()
if st.session_state.get("share_mode"):
    # Draw in-line to avoid awkward printing
    user_input = st.container().chat_input("Enter your message here.")
else:
    user_input = st.chat_input("Enter your message here.")

if user_input or st.session_state.pop("regenerate", None):
    new_msg = ConversationMessage(role="user", content=user_input)
    for c in conversations:
        c.add_message(new_msg, render=False)
    conversations[0].render_message(new_msg, container=user_msg)
    
    msg_cols = response.columns(len(conversations))
    threads = [None for _ in conversations]
    for i, conversation in enumerate(conversations):
        args = (
            conversation,
            selected_models[i],
            temperature,
            top_p,
            max_new_tokens,
            msg_cols[i],
        )
        threads[i] = threading.Thread(target=chat_response, args=args)
            
    for t in threads:
        add_script_run_ctx(t, get_script_run_ctx())
        t.start()
    for t in threads:
        t.join()
    st.rerun() # Clear stale containers

# Add action buttons

def record_feedback():
    st.toast("Feedback submitted!", icon=":material/rate_review:")

def clear_history():
    for conversation_ui in conversations:
        conversation_ui.conversation.reset_messages()
        conversation_ui.add_message(
            ConversationMessage(role="assistant", content=DEFAULT_MESSAGE),
            render=False
        )


def regenerate():
    st.session_state.regenerate = conversations[0].conversation.messages[-2].content
    for conv in conversations:
        del conv.conversation.messages[-2:]

if len(conversations[0].conversation.messages) > 1:
    feedback_cols = feedback_controls.columns(4)

    BUTTON_LABELS = [
        f"👈&nbsp; {MODEL_LABELS[0]} wins",
        f"👉&nbsp; {MODEL_LABELS[1]} wins",
        f"🤝&nbsp; Tie",
        f"👎&nbsp; Both bad",
    ]
    for i, label in enumerate(BUTTON_LABELS):
        with feedback_cols[i]:
            st.button(
                label,
                use_container_width=True,
                on_click=record_feedback,
            )


    # TODO: Big loading skeleton always briefly shows on the hosted app
    action_cols = response_controls.columns(3)

    action_cols[0].button("🔄&nbsp; Regenerate", use_container_width=True, on_click=regenerate)
    action_cols[1].button(
        "🗑&nbsp; Clear history",
        use_container_width=True,
        on_click=clear_history,
    )
    if action_cols[2].button("📷 &nbsp; Share", use_container_width=True):
        st.session_state.share_mode = True
        st.rerun()

if st.session_state.pop("share_mode", None):
    time.sleep(0.2)
    components.html(
        f"""
            <script>
            window.parent.print();
            </script>
        """,
        height=0,
        width=0,
    )