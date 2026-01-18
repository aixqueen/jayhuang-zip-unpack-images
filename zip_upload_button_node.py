from pathlib import Path
import re
import folder_paths


def _resolve_any_path(path_value: str) -> Path:
    if not path_value or not str(path_value).strip():
        raise ValueError("zip_path is empty. Use the Upload button or provide a path under input/.")
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


class ZipUploadButton:
    """A node whose UI is enhanced by frontend JS to show an Upload button."""

    @classmethod
    def INPUT_TYPES(cls):
        # NOTE: Keep this as STRING; the frontend JS will provide the upload button and write into this field.
        return {
            "required": {
                "zip_path": ("STRING", {"default": ""}),
                "validate_zip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("zip_path", "zip_name")
    FUNCTION = "run"
    CATEGORY = "utils/io"

    def run(self, zip_path, validate_zip):
        zpath = _resolve_any_path(zip_path)
        if validate_zip and zpath.suffix.lower() != ".zip":
            raise ValueError(f"Please provide a .zip file. Got: {zpath.name}")
        return (str(zip_path), zpath.name)


NODE_CLASS_MAPPINGS = {"ZipUploadButton": ZipUploadButton}
NODE_DISPLAY_NAME_MAPPINGS = {"ZipUploadButton": "Zip Upload (.zip) [Button]"}
