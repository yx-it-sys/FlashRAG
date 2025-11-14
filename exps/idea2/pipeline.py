from transformers import AutoTokenizer, AutoModelForCausalLM

class Pipeline():
    def __init__(self, model_name, prompt_path):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        