import os
import zipfile
import shutil
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IGNORE_PREFIXES = ("__MACOSX/",)


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def _resolve_any_path(path_value: str) -> Path:
    if not path_value or not str(path_value).strip():
        raise ValueError("zip_path is empty. Connect output zip_path from 'Zip Upload (.zip) [Button]'.")

    p = Path(path_value)

    if p.is_absolute() and p.exists():
        return p

    input_dir = Path(folder_paths.get_input_directory())
    cand = (input_dir / p).resolve()
    if cand.exists():
        return cand

    cand2 = p.resolve()
    if cand2.exists():
        return cand2

    raise FileNotFoundError(f"Zip path not found: {path_value}")


def _safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest_dir: Path, flatten: bool) -> Path | None:
    if member.is_dir():
        return None

    name = member.filename.replace("\\", "/")
    for pref in IGNORE_PREFIXES:
        if name.startswith(pref):
            return None

    inner = Path(name)
    out_rel = Path(inner.name) if flatten else inner

    out_path = (dest_dir / out_rel).resolve()
    dest_resolved = dest_dir.resolve()

    if not (str(out_path).startswith(str(dest_resolved) + os.sep) or out_path == dest_resolved):
        raise ValueError(f"Unsafe zip path traversal blocked: {name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(member) as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    return out_path


def _pil_to_comfy_image(pil_img: Image.Image) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


class ZipUnpackLoadFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip_path": ("STRING", {"default": ""}),

                "extract_to": (["input", "output", "temp"], {"default": "input"}),
                "subfolder": ("STRING", {"default": "unzipped_images"}),
                "clear_before_extract": ("BOOLEAN", {"default": True}),
                "flatten": ("BOOLEAN", {"default": False}),

                "max_images": ("INT", {"default": 200, "min": 1, "max": 5000, "step": 1}),
                "sort_order": (["name_asc", "name_desc"], {"default": "name_asc"}),
                "resize_to_first": ("BOOLEAN", {"default": True}),
                "auto_orient": ("BOOLEAN", {"default": True}),

                "validate_zip": ("BOOLEAN", {"default": True}),
                "max_files": ("INT", {"default": 5000, "min": 1, "max": 50000, "step": 1}),
                "max_total_mb": ("INT", {"default": 2048, "min": 1, "max": 200000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "count", "output_dir", "image_paths")
    FUNCTION = "run"
    CATEGORY = "utils/io"

    def run(
        self,
        zip_path,
        extract_to,
        subfolder,
        clear_before_extract,
        flatten,
        max_images,
        sort_order,
        resize_to_first,
        auto_orient,
        validate_zip,
        max_files,
        max_total_mb,
    ):
        zpath = _resolve_any_path(zip_path)

        if validate_zip and zpath.suffix.lower() != ".zip":
            raise ValueError(f"Please provide a .zip file. Got: {zpath.name}")

        input_dir = Path(folder_paths.get_input_directory())
        output_dir = Path(folder_paths.get_output_directory())

        file_tag = _hash_file(zpath)

        if extract_to == "input":
            root = input_dir
        elif extract_to == "output":
            root = output_dir
        else:
            root = input_dir / "_zip_temp"
            root.mkdir(parents=True, exist_ok=True)

        target_dir = root / subfolder / f"{zpath.stem}_{file_tag}"
        if clear_before_extract and target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        extracted_images: list[Path] = []

        max_total_bytes = int(max_total_mb) * 1024 * 1024
        total_unzipped_bytes = 0

        with zipfile.ZipFile(zpath, "r") as zf:
            infos = zf.infolist()

            if len(infos) > int(max_files):
                raise ValueError(f"Zip contains too many entries ({len(infos)}). Max allowed: {max_files}")

            for member in infos:
                if member.is_dir():
                    continue

                total_unzipped_bytes += int(getattr(member, "file_size", 0) or 0)
                if total_unzipped_bytes > max_total_bytes:
                    raise ValueError(f"Unzipped size exceeds limit ({max_total_mb} MB). Aborting.")

                out_path = _safe_extract_member(zf, member, target_dir, flatten=flatten)
                if out_path is None:
                    continue

                p = Path(out_path)
                if p.suffix.lower() in ALLOWED_IMAGE_EXT:
                    extracted_images.append(p)

        if not extracted_images:
            raise ValueError("No images found inside the zip (png/jpg/jpeg/webp/bmp/tif/tiff).")

        extracted_images = sorted(
            extracted_images,
            key=lambda p: p.name.lower(),
            reverse=(sort_order == "name_desc"),
        )[: int(max_images)]

        batch_tensors: list[torch.Tensor] = []
        first_size = None

        for p in extracted_images:
            img = Image.open(p)
            if auto_orient:
                img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            if first_size is None:
                first_size = img.size
            elif resize_to_first and img.size != first_size:
                img = img.resize(first_size, resample=Image.LANCZOS)

            batch_tensors.append(_pil_to_comfy_image(img))

        if not resize_to_first:
            sizes = {(t.shape[2], t.shape[1]) for t in batch_tensors}
            if len(sizes) != 1:
                raise ValueError(
                    "Images have different sizes. Enable resize_to_first=True, or ensure all images share the same resolution."
                )

        images = torch.cat(batch_tensors, dim=0)
        image_paths = "\n".join([str(p) for p in extracted_images])
        return (images, len(extracted_images), str(target_dir), image_paths)


NODE_CLASS_MAPPINGS = {"ZipUnpackLoadFromPath": ZipUnpackLoadFromPath}
NODE_DISPLAY_NAME_MAPPINGS = {"ZipUnpackLoadFromPath": "Zip Path → Unpack → Load Images"}
