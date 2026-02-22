from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

from runner import LLMRunner

class MistralRunner(LLMRunner):
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
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. "
                    "When you mention facts or sources, cite them with numbered brackets like [1], [2]. "
                    "At the end of your response, list the references as:\n"
                    "[1] Author or source title\n"
                    "[2] ...\n"
                    "Do not invent URLs."
                )
            },
            {"role": "user", "content": prompt}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        outputs = model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=False,      
            pad_token_id=tokenizer.eos_token_id
        )
 
        prompt_len = input_ids["input_ids"].shape[-1] 
        new_tokens = outputs[0][prompt_len:]    

        texto_respuesta = tokenizer.decode(new_tokens, skip_special_tokens=True)

        citas = re.findall(
            r'\[\d+\][^\[]+',
            texto_respuesta.split("References")[-1] if "References" in texto_respuesta else ""
        )
        
        return texto_respuesta, citas, tokenizer.decode(outputs, skip_special_tokens=True)