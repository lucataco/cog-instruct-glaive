from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

MODEL_NAME = "sahil2801/replit-code-instruct-glaive"
MODEL_V = "main"
MODEL_CACHE = "cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision=MODEL_V,
            cache_dir=MODEL_CACHE,
        )
        config.attn_config['attn_impl'] = 'triton'
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            config=config, 
            trust_remote_code=True,
            revision=MODEL_V,
            cache_dir=MODEL_CACHE
        )
        self.model.to(device='cuda', dtype=torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            revision=MODEL_V,
            cache_dir=MODEL_CACHE
        )

    def predict(self, 
            prompt: str = Input(description="Text prompt", default="// javascript function that returns the meaning of life"),
            max_length: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens (minimum: 1)", default=500),
            temperature: float = Input(description="For this model, 0.25 is a good starting value. (minimum: 0.01; maximum: 5)", default=0.75),
            top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens (minimum: 0.01; maximum: 1)", default=0.95),
            top_k: int = Input(description="When decoding text, samples from the top k tokens; lower to ignore less likely tokens. Defaults to 0 (no top-k sampling)", default=4),
        ) -> str:
        """Run a single prediction on the model"""
        x = self.tokenizer.encode(prompt, return_tensors='pt')
        x = x.to('cuda')
        y = self.model.generate(x, max_length=max_length, do_sample=True, top_p=top_p, top_k=top_k, pad_token_id=self.tokenizer.eos_token_id, temperature=temperature, num_return_sequences=1, eos_token_id=self.tokenizer.eos_token_id)
        generated_code = self.tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_code
