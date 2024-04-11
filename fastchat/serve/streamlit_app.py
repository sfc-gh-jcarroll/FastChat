import logging
import requests
import streamlit as st
import json
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from streamlit_messages_ui import ConversationUI
from schemas import ConversationMessage


st.set_page_config(
    page_title="Chat with Open Large Vision-Language Models",
    page_icon=":snow_capped_mountain:",
    layout="wide",
)

# Store conversation state in streamlit session
if "conversation_ui" not in st.session_state:
    st.session_state["conversation_ui"] = ConversationUI()
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
    try:
        for i, data in enumerate(streamer):
            if data["error_code"] == 0:
                output = data["text"].strip()
                chuck = conversation_ui.conversation.get_new_streaming_chuck(output)
                conversation_ui.conversation.update_streaming_msg(output)
                # conv.update_last_message(output + "▌")
                yield chuck
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                # conv.update_last_message(output)
                yield output
                return
        # output = data["text"].strip()
        # conv.update_last_message(output)
        # yield output
    except requests.exceptions.RequestException as e:
        yield f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        return
    except Exception as e:
        yield f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        return


st.title("🏔️Chat with Open Large Vision-Language Models")

# TODO: add this as command param
control_url = "http://localhost:21001"
api_endpoint_info = ""
models, all_models = get_model_list(control_url, api_endpoint_info, False)
selected_model_name = st.selectbox("Select Model", models)


# Set repetition_penalty
if "t5" in selected_model_name:
    repetition_penalty = 1.2
else:
    repetition_penalty = 1.0

conv = get_conversation_template(selected_model_name)
prompt = conv.get_prompt()

with st.expander("Parameters"):
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

conversation_ui.render_all()

user_input = st.chat_input("👉 Enter your prompt and press ENTER")

if user_input:
    conversation_ui.add_message(ConversationMessage(role="user", content=user_input))
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

        full_streamed_response = st.chat_message("Assistant").write_stream(
            stream_data(stream_iter)
        )
        conversation_ui.conversation.add_message(
            ConversationMessage(
                role="assistant", content=str(full_streamed_response).strip()
            )
        )
        conversation_ui.conversation.reset_streaming()
