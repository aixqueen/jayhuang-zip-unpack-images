# ComfyUI custom nodes entrypoint
# Exposes:
#  - Zip Upload (Button) -> returns zip_path (STRING)
#  - Zip Path -> Unpack -> Load Images -> returns IMAGE batch
#
# Also registers a small upload endpoint and a frontend JS widget.

from .zip_upload_button_node import NODE_CLASS_MAPPINGS as A, NODE_DISPLAY_NAME_MAPPINGS as AN
from .zip_unpack_load_from_path import NODE_CLASS_MAPPINGS as B, NODE_DISPLAY_NAME_MAPPINGS as BN
from .zip_upload_routes import *  # registers /ziptools/upload endpoint

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(A)
NODE_CLASS_MAPPINGS.update(B)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(AN)
NODE_DISPLAY_NAME_MAPPINGS.update(BN)

# Tell ComfyUI to load frontend JS from ./web
WEB_DIRECTORY = "./web"
