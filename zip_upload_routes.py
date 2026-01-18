import os
import time
import re
from pathlib import Path

import folder_paths

try:
    from server import PromptServer
    from aiohttp import web
except Exception:
    PromptServer = None
    web = None


def _safe_filename(name: str) -> str:
    # Keep it simple and Windows-safe
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    if not name:
        name = "upload.zip"
    return name


def _unique_name(name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    ts = int(time.time() * 1000)
    if ext:
        return f"{stem}_{ts}.{ext}"
    return f"{stem}_{ts}"


# Register endpoint only if we are inside ComfyUI runtime
if PromptServer is not None and web is not None:

    @PromptServer.instance.routes.post("/ziptools/upload")
    async def ziptools_upload(request):
        reader = await request.multipart()
        field = await reader.next()

        if field is None or field.name != "file":
            return web.json_response({"error": "missing file field"}, status=400)

        filename = _safe_filename(field.filename or "upload.zip")
        if not filename.lower().endswith(".zip"):
            return web.json_response({"error": "only .zip is allowed"}, status=400)

        # Save to ComfyUI input directory under uploaded_zips/
        input_dir = Path(folder_paths.get_input_directory())
        out_dir = input_dir / "uploaded_zips"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = _unique_name(filename)
        out_path = (out_dir / out_name).resolve()

        # Stream write
        size = 0
        max_bytes = 2 * 1024 * 1024 * 1024  # 2GB hard cap (adjust if you want)
        with open(out_path, "wb") as f:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return web.json_response({"error": "file too large"}, status=413)
                f.write(chunk)

        # Return relative path (preferred) so the node can resolve via input/
        rel_path = str(Path("uploaded_zips") / out_name).replace("\\", "/")
        return web.json_response({"ok": True, "zip_path": rel_path, "saved_as": out_name})
