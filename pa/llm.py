from abc import ABC, abstractmethod

from llama_cpp import Llama


class LLM(ABC):
    def __init__(self, model_type: str) -> None:
        self.type = model_type
        
    @abstractmethod
    def generate_response(self, query: str, config: dict = {}) -> str:
        pass


class LlamaLLM(LLM):
    
    DEFAULT_MAX_TOKENS: int = 256
    DEFAULT_TEMPERATURE: float = 0.8
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_STREAM: bool = False
    
    def __init__(self, model_path: str, n_gpu_layers:int, n_ctx:int, 
                 verbose=True) -> None:
        super().__init__('llamacpp')
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        
        self.model = Llama(model_path=model_path, 
                           verbose=self.verbose, 
                           n_gpu_layers=n_gpu_layers, 
                           n_ctx=n_ctx)
        
    def generate_response(self, query: str, config: dict={}) -> str:
        max_tokens = config['max_tokens'] if 'max_tokens' in config.keys() else self.DEFAULT_MAX_TOKENS 
        temperature = config['temperature'] if 'temperature' in config.keys() else self.DEFAULT_TEMPERATURE
        top_p = config['top_p'] if 'top_p' in config.keys() else self.DEFAULT_TOP_P
        stream = config['stream'] if 'stream' in config.keys() else self.DEFAULT_STREAM
        
        response = self.model(query,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stream=stream)
        
        response_processed = self.process_output(response, stream=stream)
        
        return response_processed
        

    def process_output(self, response: str, stream: bool = False) -> str:
        response_processed = ""
        
        if stream:
            for output in response:
                output_text = output['choices'][0]['text']
                response_processed += output_text
                print(output_text, sep='', end='', flush=True)
        else:
            response_processed = response['choices'][0]['text']
            print(response_processed)
            
        return response_processed
            