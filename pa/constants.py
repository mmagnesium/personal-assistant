import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', './db')
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', './files')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

retriever_tempate = '''### Instruction: Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. The response should be helpful and informative, but concise. 
Give the final answer in a form of the context summary with respect to the question at the end.

{context_proc}

Question: {question}

### Assistant: Let's work this out in a step by step way to be sure we have the right answer.'''

resolver_template = '''### Instruction: You are the reviewer tasked to evaluate the correctness of the provided response, based on the query and context.
Make sure that the context and response baised on it (provided below) are corrent and relevant to the query.
If the response is aligned with the question and the context then finilize it, otherwise correct it. If both the response and the context are irrelevant to the question, respond with "The response is incorrect".
Ensure that the final response is concise, helpful and factual. Dont include the context or original response in your final response, just evaluate and improve on it. 

Question: {question}

Context: {context}

Response: {response}

### Assistant: Let' work this out in a step-by-step manner, to ensure that we understant the reponse, the context, and user query correctly.'''

guidance_resolver_template = '''You are the reviewer tasked to evaluate the correctness of the provided response, based on the question and context.
Make sure that the context and response baised on it (provided below) are corrent and relevant to the query. 
Please, answer with a single word, either "Yes", "No", or "Maybe" based on the response relevance.

Question: {{question}}

Context: {{context}}

Response: {{response}}

Response review:{{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{or}} Maybe{{/select}}

Finilize the response provided above and return it in a form of concise summary. Take into account the review of the response.
If the response is irrelevant, try to correct it using the context. 
If the context is irrelevant, then answer with "The response and context are irrelevant".

Final respose: {{gen 'final_response' temperature=0.2 max_tokens=256}}
'''