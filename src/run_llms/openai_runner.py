from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from runner import LLMRunner


DEFAULT_SYSTEM_MESSAGE = """You are a concise research assistant. Respond directly and briefly. 
- After the information, add a 'References:' section listing each [n] with its full URL.
- URLs must be explicit and start with http(s). Do not list references without URLs."""


class GPTRunner(LLMRunner):
    """
    Compatible with runner.py which does:
        model_answer, references, _ = self.run_one_prompt(client, row.prompt)

    Returns:
        (answer_text, urls_citadas, raw_response)
    """

    def __init__(
        self,
        save_every: int,
        model_id: str,
        system_message: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        include_sources: bool = True,
        search_context_size: str = "low",
    ):
        super().__init__(save_every, model_id)
        self.system_message = system_message or os.getenv("GPT_SYSTEM_MESSAGE") or DEFAULT_SYSTEM_MESSAGE
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.include_sources = include_sources
        self.search_context_size = search_context_size

    def connect(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY (or GPT_API_KEY) in environment/.env")
        return OpenAI(api_key=api_key)

    def run_one_prompt(self, client: OpenAI, prompt: str):
        include = ["web_search_call.action.sources"] if self.include_sources else None

        resp = client.responses.create(
            model=self.model_id,
            input=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            tools=[{"type": "web_search"}],
            include=include,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            
        )

        answer_text = getattr(resp, "output_text", "") or ""
        urls = self._extract_urls(resp)

        return answer_text, urls, resp

    @staticmethod
    def _extract_urls(resp: Any) -> List[str]:
        """Extract URLs from url_citation annotations and (optionally) web_search sources."""

        data: Dict[str, Any] = resp.model_dump() if hasattr(resp, "model_dump") else (resp or {})

        urls: List[str] = []

        # 1) url_citation annotations on message parts
        for item in (data.get("output") or []):
            if item.get("type") != "message":
                continue
            for part in (item.get("content") or []):
                for ann in (part.get("annotations") or []):
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        urls.append(ann["url"])

        # 2) Explicit tool sources (if included)
        for item in (data.get("output") or []):
            if item.get("type") != "web_search_call":
                continue
            action = item.get("action") or {}
            for src in (action.get("sources") or []):
                url = src.get("url") or src.get("source_url") or src.get("link")
                if url:
                    urls.append(url)

        # De-duplicate, preserve order
        seen = set()
        return [u for u in urls if not (u in seen or seen.add(u))]
