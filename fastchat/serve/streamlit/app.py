import streamlit as st
from streamlit_feedback import streamlit_feedback

from messages_ui import ConversationUI
from schemas import ConversationMessage
from common import (
    get_model_list,
    get_parameters,
    chat_response,
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

    selected_model_name = st.selectbox(
        "Choose a model to chat with:", models,
        help="\n".join(f"1. **{name}:** {desc}" for name, desc in MODEL_NAMES))

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
