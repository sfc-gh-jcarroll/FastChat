import json
import logging
import requests
import streamlit as st
from messages_ui import ConversationUI
from schemas import ConversationMessage

control_url = "http://localhost:21001"
api_endpoint_info = ""

def page_setup(title, icon, wide_mode=False):
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
        layout="wide" if wide_mode else "centered",
    )

    st.title(f"{icon} {title}")

    # Add page navigation
    with st.sidebar:
        st.title("LMSYS Chatbot Arena")

        st.caption("&nbsp; &bull; &nbsp;".join([
            f"[{name}]({url})" for name, url in [
                ("Blog", "https://lmsys.org/blog/2023-05-03-arena/"),
                ("GitHub", "https://github.com/lm-sys/FastChat"),
                ("Dataset", "https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md"),
                ("Twitter", "https://twitter.com/lmsysorg"),
                ("Discord", "https://discord.gg/HSWAKCrnFx"),
            ]
        ]))

        st.write("")

        st.page_link("app.py", label="Direct Chat", icon="💬")
        st.page_link("pages/battle.py", label="Arena (battle)", icon="⚔️")
        st.page_link("pages/side_by_side.py", label="Arena (side by side)", icon="⚔️")
        st.page_link("pages/vision.py", label="Vision Direct Chat", icon="👀")
        st.page_link("pages/leaderboard.py", label="Leaderboard", icon="🏆")
        st.page_link("pages/about.py", label="About Us", icon="ℹ️")

        st.write("")
        st.write("")

        sidebar_container = st.container()

        # TOS expander

        with st.popover("Terms of Service", use_container_width=True):
            st.write("""
            **Users are required to agree to the following terms before using the service:**

            The service is a research preview. It only provides limited safety measures and may generate
            offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual
            purposes. Please do not upload any private information. The service collects user dialogue
            data, including both text and images, and reserves the right to distribute it under a
            Creative Commons Attribution (CC-BY) or a similar license.
            """)


        # Sponsors expander

        SPONSOR_LOGOS = [
            "https://storage.googleapis.com/public-arena-asset/kaggle.png",
            "https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg",
            "https://storage.googleapis.com/public-arena-asset/a16z.jpeg",
            "https://storage.googleapis.com/public-arena-asset/together.png",
            "https://storage.googleapis.com/public-arena-asset/anyscale.png",
            "https://storage.googleapis.com/public-arena-asset/huggingface.png",
        ]

        with st.popover("Sponsors", use_container_width=True):
            st.write("""
                We thank [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/),
                [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/),
                [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous
                [sponsorship](https://lmsys.org/donations/).
            """)

            st.write("") # Vertical spacing

            NUM_COLS = 3
            for i, logo in enumerate(SPONSOR_LOGOS):
                col_index = i % NUM_COLS

                if col_index == 0:
                    cols = st.columns(NUM_COLS, gap="medium")
                    st.write("") # Vertical spacing

                with cols[col_index]:
                    st.image(logo, use_column_width="auto")

    return sidebar_container


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


def encode_arctic(conversation_ui: ConversationUI):
    prompt = []
    for msg in conversation_ui.conversation.messages:
        prompt.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str

def encode_llama3(conversation_ui: ConversationUI):
    prompt = []
    prompt.append("<|begin_of_text|>")
    for msg in conversation_ui.conversation.messages:
        prompt.append(f"<|start_header_id|>{msg.role}<|end_header_id|>")
        prompt.append(f"{msg.content.strip()}<|eot_id|>")
    prompt.append("<|start_header_id|>assistant<|end_header_id|>")
    prompt.append("")
    prompt_str = "\n\n".join(prompt)
    return prompt_str

def encode_generic(conversation_ui: ConversationUI):
    prompt = []
    for msg in conversation_ui.conversation.messages:
        if msg.role == "user":
            prompt.append("user:\n" + msg.content)
        else:
            prompt.append("assistant:\n" + msg.content)

    prompt.append("assistant:")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    return prompt_str

replicate_encoding = {
    "snowflake/snowflake-arctic-instruct": encode_arctic,
    "meta/meta-llama-3-8b": encode_llama3,
    "mistralai/mistral-7b-instruct-v0.2": encode_generic,
}


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
        
        prompt_str = replicate_encoding[model_name](conversation_ui)

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
