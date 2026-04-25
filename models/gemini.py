from .base import BaseModelWrapper
import google.generativeai as genai
from config import GOOGLE_API_KEY, TEMPERATURE

class GeminiWrapper(BaseModelWrapper):
    def __init__(self, model_name="gemini-1.5-flash"):
        super().__init__(model_name, TEMPERATURE)
        if not GOOGLE_API_KEY:
            print("Warning: GOOGLE_API_KEY is not set.")
            self.model = None
        else:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Use specific model or default
            self.model = genai.GenerativeModel(model_name)
            
    def _generate(self, prompt: str) -> str | None:
        if not self.model:
            return None
            
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
            )
        )
        return response.text
