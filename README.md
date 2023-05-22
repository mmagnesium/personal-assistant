# Personal Assistant: RetirvalQA

## Description

Partially based on the [privateGPT](https://github.com/imartinez/privateGPT) project.
However, the main chain was written from scratch to speed up the inference. Not sure why LangChains implementation is so slow, like 25 seconds of prompt evaluatuion in LlamaCpp model (vs. 2-4 seconds when queried directly)

The main retrieval chain logic:

1. Given user question, the retrival agent:
   - gets the most relevant documents from the vector DB,
   - processes them, and
   - answers the question based on the context documents
2. Given the retrievers' response, the context and the question, the reviewer agent:
   - reviewes the retrievers' respond with respect to the context and user question
   - imporoves it, and
   - returns to the user

## Implementation progress

- [x] Simple retrieval QA functionality
- [ ] What is wrong with LangChain, why so slow?
- [ ] Add extra tools
- [ ] Obsidian integration?
- [ ] Docker
- [ ] ...

## Installation

Download the desired model from <https://huggingface.co/> to the `models` folder. For now, the model should be in the `ggml` format to be compatible with LlaMaCpp model interface.

Examples of tested models:

- TheBloke/wizard-mega-13B-GGML [link](https://huggingface.co/TheBloke/wizard-mega-13B-GGML)
- **TheBloke/Manticore-13B-GGML** [link](https://huggingface.co/TheBloke/Manticore-13B-GGML) - update of the wizard-lm, makes it more versitile and robust

You can use a quantized version with whatever number of bits, as long as it is in `ggml` format and supported by Llama-cpp.

### Create a python environment with Python 3.10+

**Using pyenv:**

```bash
pyenv virtualenv 3.10.9 pa
pyenv local pa
pyenv shell pa 
```

**Using conda:**

```bash
conda create -n pa python=3.10
conda activate pa
```

**Install all the requirements from the requirements.txt file:**

```bash
pip install -f requirements.txt
```

Then install `llamacpp-python` witu cuBLAS:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### Prepare the files

Create `files` folder to put the source documents in.

## Usage

Load the documents from the `files` folder to the local vector ChormaDB database:

```bash
python inject.py
```

Run the retrievalQA chain:

```bash
python retrievalQA.py
```

## References

- [Instruct Embeddings](https://arxiv.org/abs/2212.09741)
- [LlaMaCpp Python](https://github.com/abetlen/llama-cpp-python)
- [LangChain](https://github.com/hwchase17/langchain)
- [privateGPT](https://github.com/imartinez/privateGPT)
