# biohack

## Usage: 

git clone https://github.com/briandaniel14/biohack

source .venv/bin/activate (so you can use your terminal with the venv)

crtl+shift+p -> select interpreter -> biohack

pipx install uv #uv is a venv manager a bit like conda

uv sync #run this when adding new dependencies
uv run scripts/example.py

pre-commit install

git checkout -b "my_branch"