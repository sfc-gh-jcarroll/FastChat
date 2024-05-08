import requests
import streamlit as st
from streamlit_feedback import streamlit_feedback

from messages_ui import ConversationUI
from schemas import ConversationMessage
from common import (
    get_model_list,
    model_worker_stream_iter,
    page_setup,
    stream_data,
)

page_setup(
        title="LMSYS Chatbot Arena: Benchmarking LLMs in the Wild",
        icon="⚔️",
    )

# Store conversation state in streamlit session
if "side_by_side" not in st.session_state:
    st.session_state["side_by_side"] = [ConversationUI(), ConversationUI()]
    for conversation in st.session_state["side_by_side"]:
        conversation.add_message(
            ConversationMessage(role="assistant", content="Hello 👋"),
            render=False
        )
conversations = st.session_state["side_by_side"]


PROMOTION_TEXT = "[GitHub](https://github.com/lm-sys/FastChat) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx)"
MODEL_SELECT_TEXT = "**🤖 Choose any model to chat**"

st.sidebar.markdown(PROMOTION_TEXT)
# TODO: add this as command param
if st.secrets.use_arctic:
    REPLICATE_MODELS = [
        "snowflake/snowflake-arctic-instruct",
        "meta/meta-llama-3-8b",
        "mistralai/mistral-7b-instruct-v0.2",
    ]
    selected_model_name = st.sidebar.selectbox(MODEL_SELECT_TEXT, REPLICATE_MODELS)
else:
    from fastchat.model.model_adapter import (
        get_conversation_template,
    )

    control_url = "http://localhost:21001"
    api_endpoint_info = ""
    models, all_models = get_model_list(control_url, api_endpoint_info, False)
    selected_model_name = st.sidebar.selectbox(MODEL_SELECT_TEXT, models)

c = st.sidebar.popover("🔍 Model descriptions", use_container_width=True)
c0, c1, c2 = c.columns(3)
c0.markdown("Llama 3: Open foundation and chat models by Meta")
c0.markdown("Gemini: Gemini by Google")
c0.markdown("Claude: Claude by Anthropic")
c0.markdown("Phi-3: A capable and cost-effective small language models (SLMs) by Microsoft")

c1.markdown("Mixtral of experts: A Mixture-of-Experts model by Mistral AI")
c1.markdown("Reka Flash: Multimodal model by Reka")
c1.markdown("Command-R-Plus: Command-R Plus by Cohere")
c1.markdown("Command-R: Command-R by Cohere")

c2.markdown("Zephyr 141B-A35B: ORPO fine-tuned of Mixtral-8x22B-v0.1")
c2.markdown("Gemma: Gemma by Google")
c2.markdown("Qwen 1.5: A large language model by Alibaba Cloud")
c2.markdown("DBRX Instruct: DBRX by Databricks Mosaic AI")


# Set repetition_penalty
if "t5" in selected_model_name:
    repetition_penalty = 1.2
else:
    repetition_penalty = 1.0


# Parameter expander
with st.sidebar.popover("Parameters", use_container_width=True):
    temperature = st.slider(
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        label="Temperature",
    )
    top_p = st.slider(
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        label="Top P",
    )
    max_output_tokens = st.slider(
        min_value=16,
        max_value=2048,
        value=1024,
        step=64,
        label="Max output tokens",
    )
    max_new_tokens = st.slider(
        min_value=100,
        max_value=1500,
        value=1024,
        step=100,
        label="Max new tokens",
    )

tos = st.sidebar.popover("Terms of Service", use_container_width=True)
tos.markdown("""
             Users are required to agree to the following terms before using the service:

            The service is a research preview. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. Please do not upload any private information. The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license.
             """)

SPONSOR_LOGOS = [
    "https://storage.googleapis.com/public-arena-asset/kaggle.png",
    "https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg",
    "https://storage.googleapis.com/public-arena-asset/a16z.jpeg",
    "https://storage.googleapis.com/public-arena-asset/together.png",
    "https://storage.googleapis.com/public-arena-asset/anyscale.png",
    "https://storage.googleapis.com/public-arena-asset/huggingface.png",
]

with st.sidebar.popover("Sponsors", use_container_width=True):
    st.markdown("We thank [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous [sponsorship](https://lmsys.org/donations/).")
    logo_cols = st.columns(3) + st.columns(3)
    i = 0
    for logo in SPONSOR_LOGOS:
        logo_cols[i % len(logo_cols)].image(logo, use_column_width="auto")
        i += 1

# Render the chat
for c in conversations:
    c.render_all()

if user_input := st.chat_input("👉 Enter your prompt and press ENTER"):
    conversation_ui.add_message(
        ConversationMessage(role="user", content=user_input)
    )
    if st.secrets.use_arctic:
        import replicate

        prompt = []
        for msg in conversation_ui.conversation.messages:
            if msg.role == "user":
                prompt.append("<|im_start|>user\n" + msg.content + "<|im_end|>")
            else:
                prompt.append("<|im_start|>assistant\n" + msg.content + "<|im_end|>")
        
        prompt.append("<|im_start|>assistant")
        prompt.append("")
        prompt_str = "\n".join(prompt)

        model_input = {"prompt": prompt_str,
                    "prompt_template": r"{prompt}",
                    "temperature": temperature,
                    "top_p": top_p,
                    }
        stream_iter = replicate.stream(
            selected_model_name,
            input=model_input)

    else:
        ret = None
        with st.spinner("Thinking..."):
            model_api_dict = (
                api_endpoint_info[selected_model_name]
                if selected_model_name in api_endpoint_info
                else None
            )

            if model_api_dict is None:
                # Query worker address
                ret = requests.post(
                    control_url + "/get_worker_address", json={"model": selected_model_name}
                )

        if ret is not None:
            worker_addr = ret.json()["address"]

            conv = get_conversation_template(selected_model_name)
            prompt = conv.get_prompt()
            new_prompt = f"{prompt}{conversation_ui.create_new_prompt()}"

            stream_iter = model_worker_stream_iter(
                conv,
                selected_model_name,
                worker_addr,
                new_prompt,
                temperature,
                repetition_penalty,
                top_p,
                max_new_tokens,
                images=[],
            )

    full_streamed_response = st.chat_message("assistant").write_stream(
        stream_data(stream_iter, conversation_ui)
    )
    conversation_ui.conversation.add_message(
        ConversationMessage(
            role="assistant", content=str(full_streamed_response).strip()
        ),
    )
    conversation_ui.conversation.reset_streaming()

ps = st.container()

def clear_history():
    conversation_ui.conversation.reset_messages()
    conversation_ui.add_message(
        ConversationMessage(role="assistant", content="Hello 👋"),
        render=False
    )

if len(conversation_ui.conversation.messages) > 2:
    # TODO: Big loading skeleton always briefly shows on the hosted app
    cols = ps.columns(4)
    cols[0].button("⚠️ Flag", use_container_width=True)
    cols[1].button("🔄 Regenerate", use_container_width=True)
    cols[2].button(
        "🗑 Clear history",
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
            st.toast("Feedback submitted!")
