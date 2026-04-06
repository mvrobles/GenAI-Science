from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from mistralai.client import Mistral

from runner import LLMRunner

class MistralWebSearchRunner(LLMRunner):
    """
    Runner optimizado: Confía ciegamente en la configuración del Agente en la web.
    """

    def __init__(
        self,
        save_every: int,
        model_id: str,
        system_message: str | None = None,
        max_tokens: int = 1024,
    ):
        super().__init__(save_every, model_id)
        self.max_tokens = max_tokens

    def connect(self) -> Tuple[Mistral, str]:
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        # Asegúrate de que en el .env el ID sea el alfanumérico (ej: ag:7b...)
        agent_id = os.getenv("MISTRAL_AGENT_ID")
        
        if not api_key or not agent_id:
            raise RuntimeError("Falta MISTRAL_API_KEY o MISTRAL_AGENT_ID en el .env")

        client = Mistral(api_key=api_key)
        return client, agent_id

    def run_one_prompt(self, client: Tuple[Mistral, str], prompt: str):
        mistral_client, agent_id = client

        # LLAMADA MINIMALISTA: 
        # No pasamos tools, no pasamos model_id. 
        # Solo el agent_id y el mensaje.
        resp = mistral_client.agents.complete(
            agent_id=agent_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )

        answer_text = (resp.choices[0].message.content or "") if resp.choices else ""
        urls = self._extract_urls(resp)

        return answer_text, urls, resp

    @staticmethod
    def _extract_urls(resp: Any) -> List[str]:
        """Extrae URLs de las respuestas del agente (Citations/Tools)."""
        urls: List[str] = []
        # Convertimos a dict para evitar problemas de tipos del SDK
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp

        for choice in data.get("choices", []):
            message = choice.get("message", {})
            
            # Caso 1: Mistral devuelve el contenido con anotaciones (citas directas)
            content = message.get("content")
            if isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and "annotations" in chunk:
                        for ann in chunk["annotations"]:
                            u = ann.get("url") or ann.get("source_url")
                            if u: urls.append(u)
            
            # Caso 2: El agente devuelve los resultados de la herramienta de búsqueda
            for tool_call in message.get("tool_calls", []):
                try:
                    args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    # Buscamos en todas las posibles llaves de URL
                    for key in ["url", "link", "source_url"]:
                        if args.get(key): urls.append(args[key])
                    # Si devuelve una lista de fuentes
                    for src in args.get("sources", []):
                        if isinstance(src, dict):
                            u = src.get("url") or src.get("link")
                            if u: urls.append(u)
                except:
                    continue

        # Retornar lista única manteniendo orden
        return list(dict.fromkeys(urls))