from dotenv import load_dotenv
import os

from llama_cpp import Llama

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma

from constants import retriever_tempate, resolver_template

load_dotenv()

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', './db')
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', './files')

def main():
    model_path="./models/wizard-mega-13B.ggml.q4_0.bin"
    max_tokens=1024
    temperature_retrieval=0.2
    temperature_reviewer=0.2
    context_size = 4096
    
    print(f'Loading the {model_path.split("/")[-1]} model...')
    llm = Llama(model_path=model_path, 
                verbose=True, 
                n_gpu_layers=40, 
                n_ctx=context_size,
            )

    print(f'Done')
    
    print(f'Loading the embedding model...')
    embedding_instruction = "Represent the question for retrieving supporting documents "
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
                

        query = retriever_tempate.format(context_proc=context_proc, question=question)  
        # Retirval
        output_stream = llm(query,
                            max_tokens=max_tokens,
                            temperature=temperature_retrieval,
                            top_p=0.95,
                            stream=True)
        
        print('-'*20)
        print('Retrival agent response: \n\n')

        output_response = ""
        for output in output_stream:
            o_text = output['choices'][0]['text']
            output_response += o_text
            print(o_text, sep='', end='', flush=True)
            
        
        llm.reset()
    
        print('-'*20)
        
        print('Reviewer agent response: \n\n')
        
        query_reviwer = resolver_template.format(response=output_response, 
                                                 question=question,
                                                 context=context_proc)
        
        output_stream = llm(query_reviwer,
                            max_tokens=max_tokens,
                            temperature=temperature_reviewer,
                            top_p=0.95,
                            stream=True)
        
        output_response = ""
        for output in output_stream:
            o_text = output['choices'][0]['text']
            output_response += o_text
            print(o_text, sep='', end='', flush=True)
        
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