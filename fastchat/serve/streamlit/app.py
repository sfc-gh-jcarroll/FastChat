import logging
import requests
import streamlit as st
import json

from messages_ui import ConversationUI
from schemas import ConversationMessage
from streamlit_feedback import streamlit_feedback

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
    page_title="Chat with Open LLMs",
    page_icon=":snow_capped_mountain:",
)

st.title("🏔️ Chat with Open LLMs")

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
    if st.secrets.use_openai:
        for t in streamer:
            yield t
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


# TODO: add this as command param
if st.secrets.use_openai:
    selected_model_name = st.sidebar.selectbox("Select Model", ["gpt-3.5-turbo"])
else:
    from fastchat.model.model_registry import model_info
    from fastchat.constants import (
        WORKER_API_TIMEOUT,
        ErrorCode,
    )
    from fastchat.model.model_adapter import (
        get_conversation_template,
    )

    control_url = "http://localhost:21001"
    api_endpoint_info = ""
    models, all_models = get_model_list(control_url, api_endpoint_info, False)
    selected_model_name = st.sidebar.selectbox("Select Model", models)


# Set repetition_penalty
if "t5" in selected_model_name:
    repetition_penalty = 1.2
else:
    repetition_penalty = 1.0


user_input = st.chat_input("👉 Enter your prompt and press ENTER")
conversation_ui.render_all()

with st.sidebar:
    st.button("⚠️ Flag", use_container_width=True)
    st.button("🔄 Regenerate", use_container_width=True)
    st.button(
        "🗑 Clear history",
        use_container_width=True,
        on_click=conversation_ui.conversation.reset_messages,
    )

# Parameter expander
with st.sidebar.expander("Parameters"):
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


if user_input:
    conversation_ui.add_message(
        ConversationMessage(role="user", content=user_input)
    )
    if st.secrets.use_openai:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
        stream_iter = client.chat.completions.create(
            model=selected_model_name,
            messages=conversation_ui.conversation.to_openai_api_messages(),
            stream=True,
        )

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
        stream_data(stream_iter)
    )
    conversation_ui.conversation.add_message(
        ConversationMessage(
            role="assistant", content=str(full_streamed_response).strip()
        ),
    )
    conversation_ui.conversation.reset_streaming()

if len(conversation_ui.conversation.messages) > 2:
    # TODO: Big loading skeleton always briefly shows on the hosted app
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        #align="flex-start",
        key=f"feedback_{len(conversation_ui.conversation.messages)}",
    )
    if feedback:
        st.toast("Feedback submitted!")