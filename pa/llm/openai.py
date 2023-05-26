from pa.llm.base import LLM
from pa.constants import OPENAI_API_KEY

import guidance
from guidance._program import Program


class GOpenAILLM(LLM):
    def __init__(self, model='text-davinci-003') -> None:
        super().__init__('guidance_openai')
        
        self.model = guidance.llms.OpenAI(model=model,
                                          token=OPENAI_API_KEY)

        self.program: Program = None
    
    def generate_response(self, query: str, config: dict) -> str:
        guidance.llm = self.model
        self.program = guidance(query)
        self.program = self.program(**config)
        
        print(f"Response review: {self.program['answer']}")
        print(f"Finilized response: {self.program['final_response']}")
        
        return self.program.text
