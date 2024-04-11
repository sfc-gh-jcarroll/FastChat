.PHONY: install cli-vicuna-7b serve-controller serve-worker-vicuna old-ui new-ui

install:
	pip3 install -e ".[model_worker,webui]"

cli-vicuna-7b:
	python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device mps

serve-controller:
	python3 -m fastchat.serve.controller

serve-worker-vicuna:
	python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --device mps

old-ui:
	python3 -m fastchat.serve.gradio_web_server

new-ui:
	streamlit run fastchat/serve/streamlit_app.py