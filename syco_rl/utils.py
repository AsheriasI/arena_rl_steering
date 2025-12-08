import json
import re
from pathlib import Path
from typing import Any, Dict


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def load_judge_prompt(path: str = "prompts/judge_prompt.txt") -> str:
    return _resolve_path(path).read_text(encoding="utf-8").strip()


def _strip_jsonc(text: str) -> str:
    # Remove // line comments
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    # Remove /* ... */ block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text

def load_raw_prompts(path: str = "prompts/raw_prompts.jsonc") -> list[str]:
    raw = _resolve_path(path).read_text(encoding="utf-8")
    cleaned = _strip_jsonc(raw)
    return json.loads(cleaned)


def load_jsonc(path: str) -> Dict[str, Any]:
    raw = _resolve_path(path).read_text(encoding="utf-8")
    cleaned = _strip_jsonc(raw)
    return json.loads(cleaned)

