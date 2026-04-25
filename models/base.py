from abc import ABC, abstractmethod
import time
import json
import os
import hashlib
from config import CACHE_DIR

class BaseModelWrapper(ABC):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        
    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key based on model and prompt."""
        key_str = f"{self.model_name}_{self.temperature}_{prompt}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def _get_from_cache(self, prompt: str) -> str | None:
        key = self._get_cache_key(prompt)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("response")
        return None
        
    def _save_to_cache(self, prompt: str, response: str):
        key = self._get_cache_key(prompt)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"prompt": prompt, "response": response}, f)

    @abstractmethod
    def _generate(self, prompt: str) -> str | None:
        """Internal method to be implemented by subclasses."""
        pass
        
    def generate(self, prompt: str) -> str | None:
        """Public method with caching and error handling."""
        cached = self._get_from_cache(prompt)
        if cached is not None:
            return cached
            
        try:
            # Handle API rate limits implicitly, but subclass might need explicit sleep
            time.sleep(1) 
            response = self._generate(prompt)
            if response is not None:
                self._save_to_cache(prompt, response)
            return response
        except Exception as e:
            print(f"Error generating response for {self.model_name}: {e}")
            return None
