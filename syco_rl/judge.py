import asyncio
import json
import math
import os
import random
import re
import time

import httpx

from .config import JUDGE_BASE_URL, JUDGE_MODEL


class Judge:
    """
    Calls OpenRouter /chat/completions to get {\"sycophancy\": float, \"explanation\": \"...\"}.
    Per-call client/semaphore to avoid cross-loop binding.
    """
    def __init__(
        self,
        base_url: str = JUDGE_BASE_URL,
        model: str = JUDGE_MODEL,
        concurrency: int = 32,
        timeout: float = 30.0,
        api_key: str | None = None,
        referer: str | None = None,
        title: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        assert key, "Set OPENROUTER_API_KEY in your environment or pass api_key."
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self.headers = headers

    async def aclose(self):
        """No-op with per-call clients."""
        return

    def _build_body(self, system_prompt: str, user_msg: str, assistant_msg: str) -> dict:
        return {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User:\n{user_msg}\n\nAssistant:\n{assistant_msg}"},
            ],
        }

    def _backoff(self, attempt: int) -> float:
        # simple decorrelated jitter backoff: base 1s
        return (1.0 * (attempt + 1)) * (0.5 + random.random())

    async def _score_one(self, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, system_prompt: str, user_msg: str, assistant_msg: str) -> float:
        body = self._build_body(system_prompt, user_msg, assistant_msg)
        url = f"{self.base_url}/chat/completions"

        for attempt in range(3):
            try:
                async with semaphore:
                    r = await client.post(url, headers=self.headers, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code} {r.reason_phrase}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()

                txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                raw = txt.strip()

                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    print(f"[json-judge][warn] No JSON found, attempt {attempt+1}/3. Raw: {raw[:120]}")
                    await asyncio.sleep(self._backoff(attempt))
                    continue

                obj = json.loads(m.group(0))
                s = float(obj.get("sycophancy", 0.0))
                if not math.isfinite(s):
                    print(f"[json-judge][warn] Non-finite score {s}, attempt {attempt+1}/3.")
                    await asyncio.sleep(self._backoff(attempt))
                    continue

                return max(0.0, min(1.0, s))

            except httpx.HTTPStatusError as e:
                print(f"[json-judge][http-error] {e} (attempt {attempt+1}/3).")
                await asyncio.sleep(self._backoff(attempt))
            except Exception as e:
                print(f"[json-judge][error] {e} (attempt {attempt+1}/3).")
                await asyncio.sleep(self._backoff(attempt))

        print(f"[json-judge][error] All retries failed for prompt: {user_msg[:80]}... Returning 0.0 as fallback.")
        return 0.0


    async def score_batch_syco(self, system_prompt: str, user_prompts: list[str], assistant_replies: list[str]) -> list[float]:
        """
        Uses per-call client/semaphore to avoid loop binding issues.
        """
        assert len(user_prompts) == len(assistant_replies)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            semaphore = asyncio.Semaphore(self.concurrency)
            tasks = [
                self._score_one(client, semaphore, system_prompt, user_prompts[i], assistant_replies[i])
                for i in range(len(user_prompts))
            ]
            return await asyncio.gather(*tasks)

