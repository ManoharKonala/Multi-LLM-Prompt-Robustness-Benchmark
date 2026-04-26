from .base import BaseModelWrapper
from huggingface_hub import InferenceClient
from config import HF_TOKEN, TEMPERATURE

class HFWrapper(BaseModelWrapper):
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        super().__init__(model_name, TEMPERATURE)
        if not HF_TOKEN:
            print("Warning: HF_TOKEN is not set. Inference API may be rate limited.")
            self.client = InferenceClient(model=model_name)
        else:
            self.client = InferenceClient(model=model_name, token=HF_TOKEN)
            
    def _generate(self, prompt: str) -> str | None:
        if not self.client:
            return None
            
        try:
            # Use chat_completion for broader compatibility with 'Instruct' models
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=self.temperature if self.temperature > 0 else 0.01,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in HF Inference API: {e}")
            return None
