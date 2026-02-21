from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
import torch
import os

from runner import LLMRunner

class BloomRunner(LLMRunner):
    def __init__(self, save_every, model_id):
        super().__init__(save_every, model_id)

    def connect(self):
        load_dotenv()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16, 
            device_map="auto" 
        )

        return (model,tokenizer)

    def run_one_prompt(self, client, prompt):
        model, tokenizer = client
        system = (
            "You are a concise assistant. "
            "When you mention facts or sources, cite them with numbered brackets like [1], [2]. "
            "At the end of your response, list the references as:\n"
            "[1] Author or source title\n"
            "[2] ...\n"
            "Do not invent URLs."
        )
        
        full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        texto_respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        import re
        citas = re.findall(r'\[\d+\][^\[]+', texto_respuesta.split("References")[-1] if "References" in texto_respuesta else "")
        
        return texto_respuesta, citas