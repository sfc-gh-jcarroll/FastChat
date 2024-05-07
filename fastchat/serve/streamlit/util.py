import streamlit as st

def extract_diff(last_chunk, current_chunk):
    """
    Extracts the difference between the last chunk and the current chunk.

    Parameters:
    - last_chunk (str): The last received chunk of data.
    - current_chunk (str): The current chunk of data.

    Returns:
    - str: The new data present in the current chunk that was not in the last chunk.
    """
    # Find the index where the current chunk starts to differ from the last chunk
    min_len = min(len(last_chunk), len(current_chunk))
    diff_start_index = next(
        (i for i in range(min_len) if last_chunk[i] != current_chunk[i]), min_len
    )

    # Extract and return the new data from the current chunk
    new_data = current_chunk[diff_start_index:]
    return new_data

def page_setup(title, icon):
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
    )

    st.title(f"{icon} {title}")

    # Add page navigation
    with st.sidebar:
        st.title("Chatbot Arena")

        PROMOTION_TEXT = "&nbsp; &bull; &nbsp;".join([f"[{name}]({url})" for name, url in [
            ("GitHub", "https://github.com/lm-sys/FastChat"),
            ("Dataset", "https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md"),
            ("Twitter", "https://twitter.com/lmsysorg"),
            ("Discord", "https://discord.gg/HSWAKCrnFx"),
        ]])

        st.caption(PROMOTION_TEXT)

        st.write("")

        st.page_link("app.py", label="Direct Chat", icon="💬")
        st.page_link("pages/battle.py", label="Arena (battle)", icon="⚔️")
        st.page_link("pages/side_by_side.py", label="Arena (side by side)", icon="⚔️")
        st.page_link("pages/vision.py", label="Vision Direct Chat", icon="👀")
        st.page_link("pages/leaderboard.py", label="Leaderboard", icon="🏆")
        st.page_link("pages/about.py", label="About", icon="ℹ️")

        st.divider()
