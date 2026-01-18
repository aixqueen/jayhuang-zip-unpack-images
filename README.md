This is a ComfyUI custom nodes pack that makes working with ZIP files super convenient:

Upload a ZIP directly inside the node (no need to manually copy files into the input folder).

Load/unpack data from ZIP such as images, masks, prompts/text, videos, and audio for batch workflows.

Pack outputs into a single ZIP and provides a Download ZIP button right inside ComfyUI.

Includes an extra node to resize images to the nearest multiple size to avoid model dimension issues.

## Install (Git)
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/aixqueen/jayhuang-zip-unpack-images.git
```
Restart ComfyUI.

## Notes
- Uploaded zips are saved under `ComfyUI/input/uploaded_zips/`.
- For cloud (RunPod/Colab), this works the same: the browser uploads to that remote server.
