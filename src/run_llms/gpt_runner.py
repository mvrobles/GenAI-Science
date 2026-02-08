from openai import AzureOpenAI
from dotenv import load_dotenv
import os

from runner import LLMRunner

class GPTRunner(LLMRunner):
    def __init__(self, save_every, model_id):
        super().__init__(save_every, model_id)

    def connect(self):
        load_dotenv()
        api_key = os.getenv('GPT_API_KEY')
        endpoint = os.getenv('GPT_ENDPOINT')
        client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key = api_key,
            api_version="2024-10-21"
            )
        return client

def run_one_prompt(self, client, row):
    user_message = self.create_user_message(row.context, row.question, row.answer_info)

    input_messages = [
        {"role": "system", "content": self.system_message},
        {"role": "user", "content": user_message},
    ]

    resp = client.responses.create(
        model=self.model_id,
        input=input_messages,
        temperature=self.temperature,
        tools=[{"type": "web_search"}],
    )

    answer_text = resp.output_text

    data = resp.model_dump() if hasattr(resp, "model_dump") else resp
    print(data)

    urls = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("role") not in (None, "assistant"):
            continue

        for part in item.get("content", []) or []:
            for ann in part.get("annotations", []) or []:
                if ann.get("type") == "url_citation" and ann.get("url"):
                    urls.append(ann["url"])

    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    return answer_text, urls
