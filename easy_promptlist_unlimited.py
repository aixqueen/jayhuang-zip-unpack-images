from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_PROMPT_KEY_RE = re.compile(r"^prompt_(\d+)$")


class EasyPromptListUnlimited:
    """Unlimited prompt list.

    - UI starts with prompt_1.
    - Frontend JS automatically appends prompt_2, prompt_3, ... when the last field is filled.
    - Backend collects prompt_N values in numeric order and returns them as a list output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Prompt 1",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_list", "prompt_strings")

    # Both outputs are lists of strings
    OUTPUT_IS_LIST = (True, True)

    FUNCTION = "execute"
    CATEGORY = "Crush/Prompt"

    def execute(self, prompt_1: str = "", **kwargs: Any):
        prompts: List[Tuple[int, str]] = []

        def add(idx: int, val: Any):
            if not isinstance(val, str):
                return
            s = val.strip()
            if s:
                prompts.append((idx, s))

        add(1, prompt_1)

        for k, v in kwargs.items():
            m = _PROMPT_KEY_RE.match(k)
            if not m:
                continue
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            if idx <= 1:
                continue
            add(idx, v)

        prompts.sort(key=lambda x: x[0])
        out = [p for _, p in prompts]
        return (out, out)


NODE_CLASS_MAPPINGS = {
    "easy promptList (Unlimited)": EasyPromptListUnlimited,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy promptList (Unlimited)": "PromptList (Unlimited)",
}
