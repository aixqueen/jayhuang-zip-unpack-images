from __future__ import annotations

import torch
import torch.nn.functional as F


class HAIGC_ImageResizeToMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Images": ("IMAGE",),
                "multiple": (["2", "4", "8", "16", "32", "64"],),
            },
        }

    CATEGORY = "JAYHUANG/Image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"

    def _nearest_multiple(self, x: int, m: int) -> int:
        if m <= 0:
            return max(int(x), 1)
        x = max(int(x), 1)
        lower = (x // m) * m
        upper = ((x + m - 1) // m) * m
        if lower <= 0:
            lower = m
        if upper <= 0:
            upper = m
        dl = abs(x - lower)
        du = abs(upper - x)
        return upper if du <= dl else lower

    def resize(self, Images, multiple="8"):
        if Images is None:
            return (Images,)

        try:
            m = int(multiple)
        except Exception:
            m = 8

        img = Images
        if not isinstance(img, torch.Tensor):
            raise RuntimeError("Invalid Images type")

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4 or img.shape[-1] != 3:
            raise RuntimeError("Invalid Images shape")

        b, h, w, c = img.shape
        new_h = self._nearest_multiple(int(h), m)
        new_w = self._nearest_multiple(int(w), m)

        if new_h == int(h) and new_w == int(w):
            return (img,)

        x = img.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return (x,)


NODE_CLASS_MAPPINGS = {
    "JAYHUANG_ImageResizeToMultiple": HAIGC_ImageResizeToMultiple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JAYHUANG_ImageResizeToMultiple": "Resize Images to Nearest Multiple",
}

