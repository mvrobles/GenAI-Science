from dotenv import load_dotenv
from google.genai import types
from google import genai
import os

from runner import LLMRunner

class GeminiRunner(LLMRunner):
    def __init__(self, save_every, model_id):
        super().__init__(save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        client = genai.Client(api_key = api_key)
        return client

    def run_one_prompt(self, client, prompt):
        response = client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                system_instruction="You are a concise assistant. Respond directly and briefly.",
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        metadata = response.candidates[0].grounding_metadata
        urls_citadas = []

        if metadata.grounding_supports:
            indices_utilizados = []
            for support in metadata.grounding_supports:
                indices_utilizados.extend(support.grounding_chunk_indices)
            
            indices_utilizados = list(set(indices_utilizados))

            for index in indices_utilizados:
                chunk = metadata.grounding_chunks[index]
                if chunk.web:
                    urls_citadas.append(chunk.web.uri)

        return response.text, urls_citadas, response