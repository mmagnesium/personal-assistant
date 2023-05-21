from pa.agents import RetrieverAgent, ReviewerAgent
from pa.llm import LlamaLLM

        
def main():
    # Parameters
    model_path="./models/wizard-mega-13B.ggml.q4_0.bin"
    max_tokens=1024
    temperature_retrieval=0.2
    temperature_reviewer=0.2
    context_size = 4096
    stream = True
    # LLM
    print(f'Loading the {model_path.split("/")[-1]} model...')    
    llm = LlamaLLM(model_path, n_gpu_layers=40, n_ctx=context_size, verbose=True)
    print(f'Done')
    # Retriever agent
    retriever_agent = RetrieverAgent(llm, name='retriever')
    # Reviewer agent
    reviewer_agent = ReviewerAgent(llm, name='reviewver')
    # Main loop
    while True:
        # Get the question
        question = input("\nEnter a question: ")
        if question == "exit":
            break
        # Retriever chain
        retriever_response, retriever_context = retriever_agent.generate(question, 
                                                                         config={'max_tokens': max_tokens,
                                                                                 'temperature': temperature_retrieval,
                                                                                 'stream': stream})
        
        # llm.reset()
        # Reviewer chain
        reviewer_response = reviewer_agent.generate(retriever_response, question,
                                                    retriever_context, config={'max_tokens': max_tokens,
                                                                               'temperature': temperature_reviewer,
                                                                               'stream': stream})


if __name__ == "__main__":
    main()