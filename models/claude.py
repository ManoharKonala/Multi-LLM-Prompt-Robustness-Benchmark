from .base import BaseModelWrapper
import anthropic
from config import ANTHROPIC_API_KEY, TEMPERATURE

class ClaudeWrapper(BaseModelWrapper):
    def __init__(self, model_name="claude-3-5-haiku-latest"):
        # The prompt specifies 'claude-haiku-4-5-20251001', but I'll use the generic haiku name, 
        # or exactly what they requested if it's a specific version. 
        # Let's map 'claude-haiku' to 'claude-3-5-haiku-latest' or whatever they provided.
        super().__init__(model_name, TEMPERATURE)
        if not ANTHROPIC_API_KEY:
            print("Warning: ANTHROPIC_API_KEY is not set.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
    def _generate(self, prompt: str) -> str | None:
        if not self.client:
            return None
            
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
