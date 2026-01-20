from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_PROMPT_KEY_RE = re.compile(r"^prompt_(\d+)$")


class EasyPromptListUnlimited:
    """
    PromptList (Unlimited) with auto-growing INPUT PORTS (prompt_1..prompt_64).

    UI behavior (frontend JS):
      - Start by showing prompt_1 only.
      - When the last visible prompt is connected or has text, reveal the next prompt input.
      - Always keep exactly one empty prompt below the last used prompt (until reaching the max).
    """

    MAX_PROMPTS = 64

    @classmethod
    def INPUT_TYPES(cls):
        # prompt_1 required; prompt_2..prompt_MAX optional (revealed by JS)
        required = {
            "prompt_1": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "placeholder": "Prompt 1",
                },
            )
        }

        optional: Dict[str, Any] = {}
        for i in range(2, cls.MAX_PROMPTS + 1):
            optional[f"prompt_{i}"] = (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "placeholder": f"Prompt {i}",
                },
            )

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_list", "prompt_strings")

    # both outputs are lists of strings
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
            if idx <= 1 or idx > self.MAX_PROMPTS:
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
