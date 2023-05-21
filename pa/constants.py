import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', './db')
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', './files')

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

retriever_tempate = '''### Instruction: Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Try to be as precise as possible. The response should be helpful and informative, but concise. 
Try to formulate the qeustions to the context, that would help to find the answer on the question.
Dont include original context text in your response, only use your own words. Give the final answer in a form of the context summary.

{context_proc}

Question: {question}

### Assistant: Let's work this out in a step by step way to be sure we have the right answer.'''

resolver_template = '''### Instruction: You are the reviewer tasked to evaluate the correctness of the provided response, based on the query and context.
Make sure that the context and response baised on it (provided below) are corrent and relevant to the query.
If the response is not based on the context, or the context is irrelevant, reject it and respond with "The response of the retrieval agent is incorrect".
If the response and the context are correct, finilize it. Ensure that the final response is concise, helpful and factual.

Question: {question}

Context: {context}

Response: {response}

### Assistant: Let' work this out in a step-by-step manner, to ensure that we understant the reponse and user query correctly:'''
