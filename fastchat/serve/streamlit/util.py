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
