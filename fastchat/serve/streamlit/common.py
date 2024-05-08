import json
import logging
import requests
import streamlit as st
from messages_ui import ConversationUI
from schemas import ConversationMessage

control_url = "http://localhost:21001"
api_endpoint_info = ""

def page_setup(title, icon, layout="centered"):
    if "already_ran" not in st.session_state:
        st.set_option("client.showSidebarNavigation", False)
        st.session_state.already_ran = True
        st.rerun()

    # TODO: Remove from final version
    if "password" in st.secrets and "logged_in" not in st.session_state:
        passwd = st.text_input("Enter password", type="password")
        if passwd:
            if passwd == st.secrets.password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.warning("Incorrect password", icon="⚠️")
        st.stop()


    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
    )

    st.title(f"{icon} {title}")

    # Add page navigation
    with st.sidebar:
        st.subheader("Chatbot Arena")
        st.page_link("app.py", label="Direct Chat", icon="💬")
        st.page_link("pages/battle.py", label="Arena (battle)", icon="⚔️")
        st.page_link("pages/side_by_side.py", label="Arena (side by side)", icon="⚔️")
        st.page_link("pages/vision.py", label="Vision Direct Chat", icon="👀")
        st.page_link("pages/leaderboard.py", label="Leaderboard", icon="🏆")
        st.page_link("pages/about.py", label="About", icon="ℹ️")
        st.divider()

def get_model_list(controller_url, register_api_endpoint_file, multimodal):
    from fastchat.model.model_registry import model_info

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
    from fastchat.constants import WORKER_API_TIMEOUT

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


def stream_data(streamer, conversation_ui: ConversationUI):
    from fastchat.constants import ErrorCode

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


def chat_response(
        conversation_ui: ConversationUI,
        model_name: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        container=None,
    ):
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
            model_name,
            input=model_input)
    else:
        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        ret = None
        with st.spinner("Thinking..."):
            model_api_dict = (
                api_endpoint_info[model_name]
                if model_name in api_endpoint_info
                else None
            )

            if model_api_dict is None:
                # Query worker address
                ret = requests.post(
                    control_url + "/get_worker_address", json={"model": model_name}
                )

        if ret is not None:
            from fastchat.model.model_adapter import (
                get_conversation_template,
            )

            worker_addr = ret.json()["address"]

            conv = get_conversation_template(model_name)
            prompt = conv.get_prompt()
            new_prompt = f"{prompt}{conversation_ui.create_new_prompt()}"

            stream_iter = model_worker_stream_iter(
                conv,
                model_name,
                worker_addr,
                new_prompt,
                temperature,
                repetition_penalty,
                top_p,
                max_new_tokens,
                images=[],
            )
    if container:
        chat = container.chat_message("assistant")
    else:
        chat = st.chat_message("assistant")
    full_streamed_response = chat.write_stream(
        stream_data(stream_iter, conversation_ui)
    )
    conversation_ui.conversation.add_message(
        ConversationMessage(
            role="assistant", content=str(full_streamed_response).strip()
        ),
    )
    conversation_ui.conversation.reset_streaming()
