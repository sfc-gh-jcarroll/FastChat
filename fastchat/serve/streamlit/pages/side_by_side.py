import threading
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit_feedback import streamlit_feedback

from messages_ui import ConversationUI
from schemas import ConversationMessage
from common import (
    chat_response,
    get_model_list,
    page_setup,
)

sidebar_container = page_setup(
        title="LMSYS Chatbot Arena: Benchmarking LLMs in the Wild",
        icon="⚔️",
        wide_mode=True,
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


# TODO: add this as command param
if st.secrets.use_arctic:
    models = [
        "snowflake/snowflake-arctic-instruct",
        "meta/meta-llama-3-8b",
        "mistralai/mistral-7b-instruct-v0.2",
    ]
else:
    control_url = "http://localhost:21001"
    api_endpoint_info = ""
    models, all_models = get_model_list(control_url, api_endpoint_info, False)


# Sidebar

with sidebar_container:
    with st.popover("Parameters", use_container_width=True):
        temperature = st.slider(
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            label="Temperature:",
        )

        top_p = st.slider(
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            label="Top P:",
        )

        max_output_tokens = st.slider(
            min_value=16,
            max_value=2048,
            value=1024,
            step=64,
            label="Max output tokens:",
        )

        max_new_tokens = st.slider(
            min_value=100,
            max_value=1500,
            value=1024,
            step=100,
            label="Max new tokens:",
        )


# Main area

""

MODEL_NAMES = [
    ("Llama 3", "Open foundation and chat models by Meta"),
    ("Gemini", "Gemini by Google"),
    ("Claude", "Claude by Anthropic"),
    ("Phi-3", "A capable and cost-effective small language models (SLMs), by Microsoft"),
    ("Mixtral of experts", "A Mixture-of-Experts model by Mistral AI"),
    ("Reka Flash", "Multimodal model by Reka"),
    ("Command-R-Plus", "Command-R Plus by Cohere"),
    ("Command-R", "Command-R by Cohere"),
    ("Zephyr 141B-A35B", "ORPO fine-tuned of Mixtral-8x22B-v0.1"),
    ("Gemma", "Gemma by Google"),
    ("Qwen 1.5", "A large language model by Alibaba Cloud"),
    ("DBRX Instruct", "DBRX by Databricks Mosaic AI"),
]
MODEL_HELP_STR = "\n".join(f"1. **{name}:** {desc}" for name, desc in MODEL_NAMES)

selected_models = [None for _ in conversations]
model_cols = st.columns(len(selected_models))
for idx in range(len(selected_models)):
    selected_models[idx] = model_cols[idx].selectbox(
        "Choose a model to chat with:", models,
        help=MODEL_HELP_STR, key=f"model_select_{idx}")

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

if user_input := st.chat_input("👉 Enter your prompt and press ENTER"):
    new_msg = ConversationMessage(role="user", content=user_input)
    for c in conversations:
        c.add_message(new_msg, render=False)
    conversations[0].render_message(new_msg)
    
    msg_cols = st.columns(len(conversations))
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

# Add action buttons
response_controls = st.container()

def record_feedback():
    st.toast("Feedback submitted!", icon=":material/rate_review:")

def clear_history():
    for conversation_ui in conversations:
        conversation_ui.conversation.reset_messages()
        conversation_ui.add_message(
            ConversationMessage(role="assistant", content=DEFAULT_MESSAGE),
            render=False
        )

if len(conversations[0].conversation.messages) > 2:
    feedback_cols = response_controls.columns(4)

    BUTTON_LABELS = [
        f"👈&nbsp; `{selected_models[0].split('/')[-1].replace('-instruct', '')}` is better",
        f"👉&nbsp; `{selected_models[1].split('/')[-1].replace('-instruct', '')}` is better",
        f"🤝&nbsp; Tie",
        f"👎&nbsp; Both are bad",
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

    action_cols[0].button("🔄&nbsp; Regenerate", use_container_width=True)
    action_cols[1].button(
        "🗑&nbsp; Clear history",
        use_container_width=True,
        on_click=clear_history,
    )
    action_cols[2].button("📷 &nbsp; Share", use_container_width=True)
