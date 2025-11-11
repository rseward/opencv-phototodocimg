.venv:
	uv venv

install: .venv
	uv pip install -r requirements.txt

run: .venv install
	uv run phototodocimg.py

PHONY:	install run
