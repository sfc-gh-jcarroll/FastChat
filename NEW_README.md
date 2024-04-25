# Streamlit x FastChat

## Prerequisite
Install required packages
```sh
make install
```


## Instructions
To start the steamlit app

1. Start up controller
```sh
make serve-controller
```

2. Pick a model to run, this command use vicuna 7b (Run in separate terminal)
```sh
make serve-worker-vicuna
```

3. Start up streamlit app (Run in separate terminal)
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

use_openai = true

OPENAI_API_KEY = "sk-ABC123..."
```

```sh
pip install -e .
pip install --extra webui --requirement ./pyproject.toml
streamlit run fastchat/serve/streamlit/00_💬_Direct_Chat.py
```
