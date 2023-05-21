from pa.llm.base import LLM

from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LangchainLlamaLLM(LLM):
    
    DEFAULT_MAX_TOKENS: int = 256
    DEFAULT_TEMPERATURE: float = 0.8
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_STREAM: bool = True
    
    def __init__(self, model_path, n_gpu_layers:int, n_ctx:int, 
                 verbose=True) -> None:
        super().__init__('langchain_llama_cpp')
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.verbose = verbose
        
        callbacks = [StreamingStdOutCallbackHandler()]
        self.model = LlamaCpp(model_path=model_path, callbacks=callbacks,
                              n_ctx=self.n_ctx,
                              n_gpu_layers=self.n_gpu_layers,
                              verbose=self.verbose)
        
    def generate_response(self, query: str, config: dict = {}) -> str:
        max_tokens = config['max_tokens'] if 'max_tokens' in config.keys() else self.DEFAULT_MAX_TOKENS 
        temperature = config['temperature'] if 'temperature' in config.keys() else self.DEFAULT_TEMPERATURE
        top_p = config['top_p'] if 'top_p' in config.keys() else self.DEFAULT_TOP_P
        stream = config['stream'] if 'stream' in config.keys() else self.DEFAULT_STREAM
        
        self.model.max_tokens = max_tokens
        self.model.temperature = temperature
        self.model.top_p = top_p
        self.model.streaming = stream
        
        response = self.model(query)
        return response
