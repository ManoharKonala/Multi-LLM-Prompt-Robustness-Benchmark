from .base import BaseModelWrapper
from groq import Groq
from config import GROQ_API_KEY, TEMPERATURE

class GroqWrapper(BaseModelWrapper):
    def __init__(self, model_name="llama3-8b-8192"):
        super().__init__(model_name, TEMPERATURE)
        if not GROQ_API_KEY:
            print("Warning: GROQ_API_KEY is not set.")
            self.client = None
        else:
            self.client = Groq(api_key=GROQ_API_KEY)
            
    def _generate(self, prompt: str) -> str | None:
        if not self.client:
            return None
            
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=self.temperature,
        )
        return chat_completion.choices[0].message.content
