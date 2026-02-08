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
        message = client.messages.create(
            model=self.model_id, 
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
        
        final_text = ""
        
        # Extract the final text response
        for block in message.content:
            if block.type == "text":
                final_text += block.text
        
        citation_pattern = r''
        citations = re.findall(citation_pattern, final_text)
        
        # Parse document indices from citations
        doc_indices = set()
        for citation in citations:
            # Handle comma-separated citations like "0-1,2-3:5"
            parts = citation.split(',')
            for part in parts:
                # Extract just the document index (first number before '-')
                doc_index = part.split('-')[0].strip()
                try:
                    doc_indices.add(int(doc_index))
                except ValueError:
                    continue
        
        
        return final_text, list(doc_indices), message