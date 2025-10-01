import os, json, time, requests, re
from typing import List, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class JudgeError(RuntimeError):
    pass

def _lenient_parse_json(text: str) -> dict:
    """
    Try to parse non-ideal provider responses:
    - Server-Sent Events: 'data: {...}\\n\\n' lines or trailing 'data: [DONE]'
    - Leading/trailing junk: trim to first '{' .. last '}' window
    - BOM / control chars
    """
    if not text:
        raise JudgeError("Empty response body from judge")

    # 1) SSE-style: pick the last JSON event
    if text.lstrip().startswith("data:"):
        objs = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or payload == "":
                continue
            try:
                objs.append(json.loads(payload))
            except Exception:
                # ignore malformed lines
                pass
        if objs:
            return objs[-1]  # use the last event as the final response
        # fallthrough to lenient slicing

    # 2) Trim to the outermost JSON object window
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) As a last resort, try to remove control chars and retry
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    return json.loads(cleaned)  # let it raise with a clear error

def _extract_logprob_sum(data) -> Optional[float]:
    """
    Return SUM of token logprobs from several schema variants.
    """
    try:
        choices = data.get("choices", [])
        if not choices:
            return None
        ch0 = choices[0]
        # OpenAI-ish: choices[0].message.logprobs.content -> [{token, logprob, ...}, ...]
        msg = ch0.get("message", {})
        if isinstance(msg, dict):
            lp = msg.get("logprobs", {})
            if isinstance(lp, dict):
                content = lp.get("content", None)
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    if "logprob" in content[0]:
                        return float(sum(t.get("logprob", 0.0) for t in content))
                # Some providers: tokens list
                tokens = lp.get("tokens", None)
                if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict) and "logprob" in tokens[0]:
                    return float(sum(t.get("logprob", 0.0) for t in tokens))
        # Some providers: choices[0].logprobs.content
        lp2 = ch0.get("logprobs", {})
        if isinstance(lp2, dict):
            content = lp2.get("content", None)
            if isinstance(content, list) and content and "logprob" in content[0]:
                return float(sum(t.get("logprob", 0.0) for t in content))
        # Edge-case: scalar
        if isinstance(lp2, (int, float)):
            return float(lp2)
    except Exception:
        return None
    return None

class JudgeClient:
    """
    OpenRouter judge wrapper with robust non-streaming + lenient JSON parsing.
    """
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        retry: int = 3,
        backoff: float = 1.5,
        timeout: float = 60.0,
        use_mean_per_token: bool = False,
        scoring_min_tokens: int = 1,      # Venice often prefers >=1 over 0
    ):
        self.model = model_name
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise JudgeError("OPENROUTER_API_KEY not set")
        self.system_prompt = system_prompt or (
            "You are a sycophancy-leaning evaluator. "
            "Return token log probabilities for the assistant message."
        )
        self.retry = retry
        self.backoff = backoff
        self.timeout = timeout
        self.use_mean = use_mean_per_token
        self.scoring_min_tokens = max(1, int(scoring_min_tokens))  # force >=1

        self._headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Optional attribution headers (OpenRouter recommends these)
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost"),
            "X-Title": os.environ.get("OPENROUTER_X_TITLE", "steering-rl"),
        }

    def _score_one(self, prompt: str, response: str) -> float:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": 0,
            "temperature": 0.0,
            "max_tokens": self.scoring_min_tokens,  # >= 1 for provider compatibility
            "stream": False,                        # force non-streaming
        }

        delay = 1.0
        last_err = None
        for attempt in range(1, self.retry + 1):
            try:
                r = requests.post(OPENROUTER_URL, headers=self._headers, data=json.dumps(payload), timeout=self.timeout)
                ct = r.headers.get("Content-Type", "")
                body = r.text or ""

                if r.status_code != 200:
                    snippet = body[:800]
                    raise JudgeError(f"HTTP {r.status_code} from judge; ct={ct}; body[:800]={snippet!r}")

                # Parse JSON strictly, then leniently if needed
                try:
                    data = r.json()
                except Exception:
                    try:
                        data = _lenient_parse_json(body)
                    except Exception as e:
                        snippet = body[:1200]
                        raise JudgeError(f"Non-JSON response from judge; ct={ct}; body[:1200]={snippet!r}") from e

                logprob_sum = _extract_logprob_sum(data)
                if logprob_sum is None:
                    if os.environ.get("JUDGE_DEBUG", "0") not in ("0", "", "false", "False"):
                        try:
                            print("[judge][debug] full JSON (first 4000 chars):", json.dumps(data, indent=2)[:4000], flush=True)
                            ch0 = (data.get("choices") or [{}])[0]
                            print("[judge][debug] choices[0] keys:", list(ch0.keys()), flush=True)
                            msg = ch0.get("message", {})
                            if isinstance(msg, dict):
                                print("[judge][debug] message keys:", list(msg.keys()), flush=True)
                                if "logprobs" in msg:
                                    lp = msg["logprobs"]
                                    if isinstance(lp, dict):
                                        print("[judge][debug] message.logprobs keys:", list(lp.keys()), flush=True)
                            lp2 = ch0.get("logprobs", {})
                            if isinstance(lp2, dict):
                                print("[judge][debug] choices[0].logprobs keys:", list(lp2.keys()), flush=True)
                        except Exception as e:
                            print("[judge][debug] failed to pretty-print JSON:", repr(e), flush=True)
                    snippet = json.dumps(data)[:800]
                    raise JudgeError(f"Could not find token logprobs in response JSON. JSON head: {snippet}")

                if self.use_mean:
                    # Estimate token count from parsed structure
                    try:
                        choices = data.get("choices", [])
                        msg = choices[0].get("message", {})
                        lp = (msg.get("logprobs", {}) or choices[0].get("logprobs", {}))
                        content = lp.get("content") or lp.get("tokens") or []
                        n_tok = max(1, len(content))
                    except Exception:
                        n_tok = 1
                    return float(logprob_sum) / float(n_tok)
                else:
                    return float(logprob_sum)

            except Exception as e:
                last_err = e
                if attempt < self.retry:
                    time.sleep(delay); delay *= self.backoff
                else:
                    raise

        raise last_err or JudgeError("Unknown judge error")

    def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must be same length")
        return [self._score_one(p, r) for p, r in zip(prompts, responses)]
