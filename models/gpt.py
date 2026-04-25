from .base import BaseModelWrapper
from openai import OpenAI
from config import OPENAI_API_KEY, TEMPERATURE

class GPTWrapper(BaseModelWrapper):
    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__(model_name, TEMPERATURE)
        if not OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY is not set.")
            self.client = None
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            
    def _generate(self, prompt: str) -> str | None:
        if not self.client:
            return None
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content
