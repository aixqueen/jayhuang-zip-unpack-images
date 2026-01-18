# ComfyUI Zip Tools (Upload + Unpack + Load)

Two nodes for a clean workflow:

1) **Zip Upload Only (.zip)**
   - Upload a `.zip` directly in the node.
   - Outputs `zip_path` (STRING).

2) **Zip Path → Unpack → Load Images**
   - Takes `zip_path` (STRING) from node 1.
   - Extracts safely (zip-slip protection) with limits to avoid zip-bombs.
   - Loads images and outputs `IMAGE` batch.

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/aixqueen/jayhuang-zip-unpack-images.git
```

Restart ComfyUI.

## Folder layout

```
ComfyUI/custom_nodes/comfyui-zip-tools/
  __init__.py
  zip_upload_only.py
  zip_unpack_load_from_path.py
  .gitignore
  README.md
```

## Notes

- IMAGE batches require identical H/W. Keep `resize_to_first=True` unless your images are already the same size.
- If your UI build does not show a file picker for `FILE`, upload the zip into `input/` and select it by name.
