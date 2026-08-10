from transformers import pipeline


class ModelManager:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False

    def _load(self):
        """Load the model pipeline."""
        if self._loaded:
            return

        print(f"🔄 Loading model: {self.model_name}...")

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            trust_remote_code=True,
        )

        self._loaded = True
        print(f"✅ Model loaded")

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 256
    ) -> str:
        """Generate text from prompt."""
        if not self._loaded:
            self._load()
        result = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        full = result[0]["generated_text"]
        if prompt in full:
            return full[len(prompt) :].strip()
        return full.strip()
