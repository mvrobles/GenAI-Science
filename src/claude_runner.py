from dotenv import load_dotenv
import anthropic
import os

from runner import LLMRunner

class ClaudeRunner(LLMRunner):
    def __init__(self, temperature, save_every, model_id):
        super().__init__(temperature, save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = anthropic.Anthropic(api_key=api_key)

        return client

    def run_one_prompt(self, client, prompt):
        message = client.messages.create(
            model="claude-sonnet-4-20250514", 
            max_tokens=1024,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search"
                }
            ],
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
            
        return message.content[0].text
