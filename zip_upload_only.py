from pathlib import Path
import folder_paths


def _resolve_uploaded_path(file_value: str) -> str:
    input_dir = Path(folder_paths.get_input_directory())
    p = Path(file_value)

    if p.is_absolute() and p.exists():
        return str(p)

    cand = (input_dir / p).resolve()
    if cand.exists():
        return str(cand)

    cand2 = p.resolve()
    if cand2.exists():
        return str(cand2)

    raise FileNotFoundError(f"File not found: {file_value}")


class ZipUploadOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip_file": ("FILE", {"default": ""}),
                "validate_zip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("zip_path", "zip_name")
    FUNCTION = "run"
    CATEGORY = "utils/io"

    def run(self, zip_file, validate_zip):
        zpath = _resolve_uploaded_path(zip_file)
        if validate_zip and not zpath.lower().endswith(".zip"):
            raise ValueError(f"Please upload a .zip file. Got: {Path(zpath).name}")
        return (zpath, Path(zpath).name)


NODE_CLASS_MAPPINGS = {"ZipUploadOnly": ZipUploadOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"ZipUploadOnly": "Zip Upload Only (.zip)"}
