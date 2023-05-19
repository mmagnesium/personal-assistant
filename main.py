from dotenv import load_dotenv
import os

from llama_cpp import Llama


from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma

from langchain.agents import initialize_agent, AgentType

# import streamlit as st

load_dotenv()

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', './files')

def main():
    model_path="./models/wizard-mega-13B.ggml.q4_0.bin"
    max_tokens=512
    temperature=0.2
    context_size = 2048
    
    print(f'Loading the {model_path.split("/")[-1]} model...')
    llm = Llama(model_path=model_path, 
                verbose=True, 
                n_gpu_layers=40, 
                n_ctx=context_size,
            )
    
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # llm = LlamaCpp(
    #     model_path=model_path, 
    #     callback_manager=callback_manager, 
    #     verbose=True,
    #     n_gpu_layers=40,
    #     n_ctx=context_size,
    #     max_tokens=max_tokens,
    #     temperature=temperature
    # )
    print(f'Done')
    
    print(f'Loading the embedding model...')
    embedding_instruction = "Represent the question for retrieving supporting documents; Input: "
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                          model_kwargs={"device": "cuda"},
                                                          query_instruction=embedding_instruction)
    print(f'Done')
    
    print(f'Loading the vector database retriever client...')
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=instructor_embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    print(f'Done\n')

    while True:
        question = input("\nEnter a question: ")
        if question == "exit":
            break
    
        context_docs = [doc for doc in retriever.get_relevant_documents(question)]
        context_proc = ""
        context_included = []

        for i, doc in enumerate(context_docs):
            if context_proc.find(doc.page_content) == -1:
                context_included.append(True)
                context_proc += f'Source: {i}' + doc.metadata['source'] + "\n"
                context_proc += 'Information: ' + doc.page_content + '\n' 
            else:
                context_included.append(False)

        query = f'''### Instruction: Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Try to be as precise as possible. The response should be helpful and informative, but concise. 
        Dont include original context text in your response, only use your own words. Give the final answer in a form of the context summary.

        {context_proc}

        Question: {question}

        ### Assistant: Let's work this out in a step by step way to be sure we have the right answer.'''
        
        # prompt_template = PromptTemplate(template=template, input_variables=["context_proc", "question"])
        # llm_chain = LLMChain(prompt=prompt_template, 
        #                      llm=llm)
        
        # res = llm_chain.run(question=query, context_proc=context_proc)
        
        output_stream = llm(query,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=0.95,
                            stream=True)

        print('-'*20)

        output_response = ""
        for output in output_stream:
            o_text = output['choices'][0]['text']
            output_response += o_text
            print(o_text, sep='', end='', flush=True)
        
        # Print the result
        # print("\n\n> Question:")
        # print(query)
        # print("\n> Answer:")
        # print(res.strip('\n'))
        
        print('-'*20)
        
        print(f'Sources: \n\n')
        for i, doc in enumerate(context_docs):
            if context_included[i]:
                print(f'Source: {i}' + doc.metadata['source'])
                print('Context: ' + doc.page_content)
                print('-'*10 + '\n')
        print('End of chain.')
        print('-'*20)


if __name__ == "__main__":
    main()