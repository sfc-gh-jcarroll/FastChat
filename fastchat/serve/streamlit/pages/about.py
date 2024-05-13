import streamlit as st

from common import page_setup, SPONSOR_LOGOS

page_setup(
    title="About Us",
    icon="ℹ️",
)

st.markdown("""
Chatbot Arena is an open-source research project developed by members from [LMSYS](https://lmsys.org) and UC Berkeley [SkyLab](https://sky.cs.berkeley.edu/). Our mission is to build an open platform to evaluate LLMs by human preference in the real-world.
We open-source our [FastChat](https://github.com/lm-sys/FastChat) project at GitHub and release chat and human feedback dataset. We invite everyone to join us!

### Arena Core Team
- [Lianmin Zheng](https://lmzheng.net/) (co-lead), [Wei-Lin Chiang](https://infwinston.github.io/) (co-lead), [Ying Sheng](https://sites.google.com/view/yingsheng/home), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/), [Ion Stoica](http://people.eecs.berkeley.edu/~istoica/)

### Past Members
- [Siyuan Zhuang](https://scholar.google.com/citations?user=KSZmI5EAAAAJ), [Hao Zhang](https://cseweb.ucsd.edu/~haozhang/)

### Learn more
- Chatbot Arena [paper](https://arxiv.org/abs/2403.04132), [launch blog](https://lmsys.org/blog/2023-05-03-arena/), [dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md), [policy](https://lmsys.org/blog/2024-03-01-policy/)
- LMSYS-Chat-1M dataset [paper](https://arxiv.org/abs/2309.11998), LLM Judge [paper](https://arxiv.org/abs/2306.05685)

### Contact Us
- Follow our [X](https://x.com/lmsysorg), [Discord](https://discord.gg/HSWAKCrnFx) or email us at lmsys.org@gmail.com
- File issues on [GitHub](https://github.com/lm-sys/FastChat)
- Download our datasets and models on [HuggingFace](https://huggingface.co/lmsys)

### Acknowledgment
We thank [SkyPilot](https://github.com/skypilot-org/skypilot) and [Gradio](https://github.com/gradio-app/gradio) team for their system support.
We also thank [UC Berkeley SkyLab](https://sky.cs.berkeley.edu/), [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Hyperbolic](https://hyperbolic.xyz/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous sponsorship. Learn more about partnership [here](https://lmsys.org/donations/).
            """)

NUM_COLS = 4
for i, logo in enumerate(SPONSOR_LOGOS):
    col_index = i % NUM_COLS

    if col_index == 0:
        cols = st.columns(NUM_COLS, gap="medium")
        st.write("") # Vertical spacing

    with cols[col_index]:
        st.image(logo, use_column_width="auto")
