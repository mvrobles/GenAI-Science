from dotenv import load_dotenv
import anthropic
import os
import re

from runner import LLMRunner

class ClaudeRunner(LLMRunner):
    def __init__(self, save_every, model_id):
        super().__init__(save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = anthropic.Anthropic(api_key=api_key)

        return client

    def run_one_prompt(self, client, prompt):
        response = client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search"
            }],
            system="You are a concise assistant. Respond directly and briefly.",
            messages=[{"role": "user", "content": prompt}]
        )

        urls_citadas = []
        texto_respuesta = ""

        for block in response.content:
            if block.type == "text":
                texto_respuesta += block.text
                if hasattr(block, 'citations') and block.citations:
                    for citation in block.citations:
                        url = getattr(citation, 'url', None)
                        if url:
                            urls_citadas.append(url)

        urls_citadas = list(set(urls_citadas))

        return texto_respuesta, urls_citadas, response