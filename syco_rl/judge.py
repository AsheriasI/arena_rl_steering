import asyncio
import json
import math
import os
import re

import httpx

from .config import JUDGE_BASE_URL, JUDGE_MODEL, accelerator


class LocalVLLMJSONJudge:
    """
    Calls /v1/chat/completions to get {"sycophancy": float, "explanation": "..."}.
    [FIXED (Comment 2)] Now uses a persistent client and includes retries.
    """
    def __init__(self, base_url=JUDGE_BASE_URL, model=JUDGE_MODEL, concurrency: int = 32, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}

        # OpenRouter requires the bearer token; referer/title are recommended but optional
        api_key = os.environ.get("OPENROUTER_API_KEY")
        assert api_key, "Set OPENROUTER_API_KEY in your environment."
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional (helps with routing/usage analytics):
            # "HTTP-Referer": "http://localhost",    # or your site/app URL
            # "X-Title": "sycophancy-judge",
        }
        
        # [FIX (Comment 2)] Create a persistent client
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self):
        """Closes the httpx client."""
        if accelerator.is_main_process:
            print("[judge] Closing HTTP client...")
        await self.client.aclose()

    async def _score_one(self, client: httpx.AsyncClient, system_prompt: str, user_msg: str, assistant_msg: str) -> float:
        """
        [FIXED (Comment 2)] Added retry logic for network/judge errors.
        """
        body = {
            "model": self.model,
            "temperature": 0.0,
            # "max_tokens": None,
            "response_format": {"type": "json_object"},  # ensure we get text back
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User:\n{user_msg}\n\nAssistant:\n{assistant_msg}"},
            ]
        }
        
        # [FIX (Comment 2)] Add retry logic
        for attempt in range(3): # Try 3 times
            try:
                async with self.semaphore:
                    r = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=body)
                    r.raise_for_status()
                    data = r.json()

                # --- DEBUG: show raw completion text ---
                txt = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                raw = txt.strip()
                
                # Try to extract JSON
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    print(f"[json-judge][warn] No JSON found, attempt {attempt+1}/3. Retrying... Raw: {raw[:100]}")
                    await asyncio.sleep(1.0 * (attempt + 1)) # Exponential backoff
                    continue # Go to next attempt

                obj = json.loads(m.group(0))
                s = float(obj.get("sycophancy", 0.0))
                
                if not math.isfinite(s):
                    print(f"[json-judge][warn] Non-finite score {s}, attempt {attempt+1}/3. Retrying...")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue # Go to next attempt

                return max(0.0, min(1.0, s)) # Success!

            except httpx.HTTPStatusError as e:
                print(f"[json-judge][http-error] {e} (attempt {attempt+1}/3). Retrying...")
                await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as e:
                print(f"[json-judge][error] {e} (attempt {attempt+1}/3). Retrying...")
                await asyncio.sleep(1.0 * (attempt + 1))

        print(f"[json-judge][error] All retries failed for prompt: {user_msg[:50]}... Returning 0.0 as fallback.")
        return 0.0 # Fallback after all retries fail


    async def score_batch_syco(self, system_prompt: str, user_prompts: list[str], assistant_replies: list[str]) -> list[float]:
        """
        [FIXED (Comment 2)] Uses the persistent self.client.
        """
        assert len(user_prompts) == len(assistant_replies)
        # [FIX (Comment 2)] Use self.client instead of creating a new one
        tasks = [
            self._score_one(self.client, system_prompt, user_prompts[i], assistant_replies[i])
            for i in range(len(user_prompts))
        ]
        return await asyncio.gather(*tasks)

