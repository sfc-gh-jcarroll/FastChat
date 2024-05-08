import streamlit as st
from streamlit_feedback import streamlit_feedback

from messages_ui import ConversationUI
from schemas import ConversationMessage
from common import (
    get_model_list,
    chat_response,
    page_setup,
)
from util import page_setup

page_setup(
        title="Chat with Open LLMs",
        icon="🏔️",
    )

# Store conversation state in streamlit session
if "conversation_ui" not in st.session_state:
    st.session_state["conversation_ui"] = ConversationUI()
    st.session_state["conversation_ui"].add_message(
        ConversationMessage(role="assistant", content="Hello 👋"),
        render=False
    )
conversation_ui: ConversationUI = st.session_state["conversation_ui"]


def get_model_list(controller_url, register_api_endpoint_file, multimodal):
    global api_endpoint_info

    # Add models from the controller
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200

        if multimodal:
            ret = requests.post(controller_url + "/list_multimodal_models")
            models = ret.json()["models"]
        else:
            ret = requests.post(controller_url + "/list_language_models")
            models = ret.json()["models"]
    else:
        models = []

    # Add models from the API providers
    if register_api_endpoint_file:
        api_endpoint_info = json.load(open(register_api_endpoint_file))
        for mdl, mdl_dict in api_endpoint_info.items():
            mdl_multimodal = mdl_dict.get("multimodal", False)
            if multimodal and mdl_multimodal:
                models += [mdl]
            elif not multimodal and not mdl_multimodal:
                models += [mdl]

    # Remove anonymous models
    models = list(set(models))
    visible_models = models.copy()
    for mdl in visible_models:
        if mdl not in api_endpoint_info:
            continue
        mdl_dict = api_endpoint_info[mdl]
        if mdl_dict["anony_only"]:
            visible_models.remove(mdl)

    # Sort models and add descriptions
    priority = {k: f"___{i:03d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    visible_models.sort(key=lambda x: priority.get(x, x))
    return visible_models, models


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
    images,
):
    # Make requests
    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    logging.info(f"==== request ====\n{gen_params}")

    if len(images) > 0:
        gen_params["images"] = images

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )

    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def stream_data(streamer):
    if st.secrets.use_arctic:
        for t in streamer:
            yield str(t)
    try:
        for i, data in enumerate(streamer):
            if data["error_code"] == 0:
                output = data["text"].strip()
                chuck = conversation_ui.conversation.get_new_streaming_chuck(output)
                conversation_ui.conversation.update_streaming_msg(output)
                yield chuck
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                yield output
                return
    except requests.exceptions.RequestException as e:
        yield f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        return
    except Exception as e:
        yield f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        return


MODEL_SELECT_TEXT = "**🤖 Choose any model to chat**"

# TODO: add this as command param
if st.secrets.use_arctic:
    REPLICATE_MODELS = [
        "snowflake/snowflake-arctic-instruct",
        "meta/meta-llama-3-8b",
        "mistralai/mistral-7b-instruct-v0.2",
    ]
    selected_model_name = st.sidebar.selectbox(MODEL_SELECT_TEXT, REPLICATE_MODELS)
else:
    control_url = "http://localhost:21001"
    api_endpoint_info = ""
    models, all_models = get_model_list(control_url, api_endpoint_info, False)
    selected_model_name = st.sidebar.selectbox(MODEL_SELECT_TEXT, models)


with st.sidebar:
    with st.popover("🔍 Model descriptions", use_container_width=True):
        c0, c1, c2 = st.columns(3)

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
with st.sidebar:
    with st.popover("Parameters", use_container_width=True):
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

with st.sidebar:
    with st.popover("Terms of Service", use_container_width=True):
        st.markdown("""
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

with st.sidebar:
    with st.popover("Sponsors", use_container_width=True):
        st.markdown("We thank [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous [sponsorship](https://lmsys.org/donations/).")
        logo_cols = st.columns(3) + st.columns(3)
        i = 0
        for logo in SPONSOR_LOGOS:
            logo_cols[i % len(logo_cols)].image(logo, use_column_width="auto")
            i += 1

# Render the chat
conversation_ui.render_all()

if user_input := st.chat_input("👉 Enter your prompt and press ENTER"):
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
