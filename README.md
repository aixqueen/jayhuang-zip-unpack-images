# ComfyUI Zip Tools (True Upload Button)

This extension provides a real **Upload .zip** button inside the node UI.

## Nodes
1) **Zip Upload (.zip) [Button]**
   - Has an Upload button (frontend JS).
   - Uploads to ComfyUI server endpoint `/ziptools/upload`.
   - Outputs `zip_path` (STRING) relative to `input/`.

2) **Zip Path → Unpack → Load Images**
   - Takes `zip_path` and outputs `IMAGE` batch.

## Install (Git)
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/aixqueen/jayhuang-zip-unpack-images.git
```
Restart ComfyUI.

## Notes
- Uploaded zips are saved under `ComfyUI/input/uploaded_zips/`.
- For cloud (RunPod/Colab), this works the same: the browser uploads to that remote server.
