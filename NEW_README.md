# Streamlit x FastChat

## Prerequisite

Install required packages

```sh
make install
```

## Instructions

To start the steamlit app

(1) Start up controller

```sh
make serve-controller
```

(2) Pick a model to run, this command use vicuna 7b (Run in separate terminal)

```sh
make serve-worker-vicuna
```

(3) Start up streamlit app (Run in separate terminal)

```sh
make new-ui
```

if you want to try the gradio version of UI, use

```sh
make old-ui
```

## Simple prototyping version

```toml
# .streamlit/secrets.toml

use_arctic = true

REPLICATE_API_TOKEN = "r8_ABC123..."

password = "PASS"
```

```sh
pip install -r fastchat/serve/streamlit/requirements.txt
streamlit run fastchat/serve/streamlit/app.py
```

### Leaderboard

Download the latest version of the following files:

- `elo_results_YYYYMMDD.pkl`
- `leaderboard_table_YYYYMMDD.csv`

from [here](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/tree/main).
