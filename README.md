# Personal Assistant: RetirvalQA

## Installation

Download the desired model from <https://huggingface.co/TheBloke/wizard-mega-13B-GGML> to the `models` folder.

Create a python environment with Python 3.10

Install all the requirements from the requirements.txt file.

```bash
pip install -f requirements.txt
```

Create `files` folder to put the desired documents to query.

## Usage

- `inject.py` - load the documents from the `files` folder to the local vector ChormaDB database.

- `main.py` - run the main querying loop of the program.

## Instruct Embeddings

Paper: [link](https://arxiv.org/abs/2212.09741)
