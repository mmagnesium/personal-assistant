from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self, model_type: str) -> None:
        self.type = model_type
        
    @abstractmethod
    def generate_response(self, query: str) -> str:
        pass
