from __future__ import annotations

"""
Ollama LLM Provider.

Connects to a local Ollama instance for language model inference.
Supports any model available in Ollama (Llama, Qwen, Mistral, etc.)
"""

import json
import logging
import re
import time
from typing import AsyncIterator

import httpx

from src.core.interfaces.llm import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class _ThinkTagFilter:
    """
    Incrementally strips <think>...</think> blocks from a token stream.

    Tags can be split across chunk boundaries, so a small tail is held
    back until it can't be the start of a tag. Without this, models that
    ignore think:false leak reasoning into the display and TTS.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        """Feed a raw chunk, get back the speakable text (may be empty)."""
        self._buf += chunk
        out = ""
        while True:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    # Keep only a tail that could still be a partial close tag
                    self._buf = self._buf[-(len(self._CLOSE) - 1):]
                    break
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_think = False
            else:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    # Emit everything except a suffix that could be a partial open tag
                    keep = 0
                    max_k = min(len(self._OPEN) - 1, len(self._buf))
                    for k in range(max_k, 0, -1):
                        if self._buf.endswith(self._OPEN[:k]):
                            keep = k
                            break
                    cut = len(self._buf) - keep
                    out += self._buf[:cut]
                    self._buf = self._buf[cut:]
                    break
                out += self._buf[:idx]
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_think = True
        return out

    def flush(self) -> str:
        """Return any held-back text at end of stream (dropped if mid-think)."""
        out = "" if self._in_think else self._buf
        self._buf = ""
        return out


class OllamaLLMProvider(LLMProvider):
    """
    LLM Provider using Ollama's REST API.

    Swap models by changing config — no code changes needed.
    """

    def __init__(self):
        self.model: str = "llama3.1:8b"
        self.base_url: str = "http://localhost:11434"
        self.think: bool = False
        self.keep_alive: int | str = -1
        self.client: httpx.AsyncClient | None = None

    async def initialize(self, config: dict) -> None:
        self.model = config.get("model", "llama3.1:8b")
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.think = config.get("think", False)
        # -1 = keep the model loaded indefinitely. Sent per-request so it
        # works regardless of how the Ollama server is configured/installed
        # (the OLLAMA_KEEP_ALIVE env var never reaches the desktop app).
        self.keep_alive = config.get("keep_alive", -1)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(600.0, connect=10.0),  # 10min for large models
        )

        # Verify Ollama is running and model is available
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

            # Check if model is pulled (handle tag variations)
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)

            if not available:
                logger.warning(
                    f"Model '{self.model}' may not be available. "
                    f"Pull it with: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama connected. Model: {self.model}")
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama. Is it running? "
                "Start with: ollama serve"
            )
            raise

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start_time = time.time()

        # Build messages list with system prompt
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "think": self.think,  # Configurable thinking mode
            "keep_alive": self.keep_alive,  # -1 = never unload (avoid cold starts mid-show)
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = await self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw_text = data.get("message", {}).get("content", "")
        generation_time = (time.time() - start_time) * 1000

        # Strip any leaked thinking blocks, then parse emotion tag
        raw_text = self._strip_thinking(raw_text)
        emotion, clean_text = self._parse_emotion(raw_text)

        # Calculate tokens
        eval_count = data.get("eval_count", 0)

        return LLMResponse(
            text=clean_text,
            emotion=emotion,
            generation_time_ms=generation_time,
            tokens_generated=eval_count,
        )

    async def stream_generate_with_metadata(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str | LLMResponse]:
        """
        Stream text chunks, then yield a final LLMResponse with metadata.

        Yields:
            str chunks as they arrive, then a single LLMResponse as the last item.
        """
        start_time = time.time()
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": True,
            "think": self.think,  # Configurable thinking mode
            "keep_alive": self.keep_alive,  # -1 = never unload (avoid cold starts mid-show)
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        accumulated_text = ""
        eval_count = 0
        first_token_time = None
        think_filter = _ThinkTagFilter()

        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            clean = think_filter.feed(content)
                            if clean:
                                accumulated_text += clean
                                yield clean
                        # Capture token count from the final chunk
                        if data.get("done", False):
                            eval_count = data.get("eval_count", 0)
                    except json.JSONDecodeError:
                        continue

        tail = think_filter.flush()
        if tail:
            accumulated_text += tail
            yield tail

        # After streaming completes, parse and yield final metadata
        generation_time = (time.time() - start_time) * 1000
        ttft = ((first_token_time - start_time) * 1000) if first_token_time else generation_time
        accumulated_text = self._strip_thinking(accumulated_text)
        emotion, clean_text = self._parse_emotion(accumulated_text)

        yield LLMResponse(
            text=clean_text,
            emotion=emotion,
            generation_time_ms=generation_time,
            first_token_time_ms=ttft,
            tokens_generated=eval_count,
        )

    async def get_model_info(self) -> dict:
        resp = await self.client.post(
            "/api/show", json={"name": self.model}
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "name": self.model,
            "parameters": data.get("details", {}).get("parameter_size", "unknown"),
            "family": data.get("details", {}).get("family", "unknown"),
            "quantization": data.get("details", {}).get("quantization_level", "unknown"),
        }

    async def shutdown(self) -> None:
        if self.client:
            # keep_alive: -1 pins the model in RAM for the whole run —
            # explicitly unload it on clean shutdown so quitting the app
            # frees the memory (keep_alive: 0 = unload now)
            try:
                await self.client.post(
                    "/api/generate",
                    json={"model": self.model, "keep_alive": 0},
                    timeout=10.0,
                )
                logger.info(f"Ollama model {self.model} unloaded")
            except Exception as e:
                logger.debug(f"Could not unload model on shutdown: {e}")
            await self.client.aclose()

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Strip Qwen 3.5 thinking blocks from response.

        Qwen 3.5 wraps internal reasoning in <think>...</think> tags.
        We remove these so only the spoken response remains.
        """
        # Remove <think>...</think> blocks (including multiline)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def _parse_emotion(text: str) -> tuple[str | None, str]:
        """
        Parse emotion tag from LLM response.

        Only matches a tag at the START of the text. Brackets later in
        the reply ("Hola mamá [risas] qué bueno verte") are spoken
        content — matching them would silently delete dialogue.

        Input:  "[warm, nostalgic] I remember when..."
        Output: ("warm, nostalgic", "I remember when...")
        """
        match = re.match(r"\s*\[([^\]]+)\]\s*(.*)", text, re.DOTALL)
        if match:
            emotion = match.group(1).strip()
            clean_text = match.group(2).strip()
            return emotion, clean_text
        return None, text.strip()
