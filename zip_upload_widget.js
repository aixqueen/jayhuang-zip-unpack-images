// ComfyUI frontend extension:
// Adds an Upload button to the node "Zip Upload (.zip) [Button]"
// Uploads the selected zip to /ziptools/upload and writes the returned path into the zip_path widget.

import { app } from "../../scripts/app.js";

function findWidget(node, name) {
  if (!node.widgets) return null;
  return node.widgets.find(w => w.name === name) || null;
}

async function uploadZip(file) {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch("/ziptools/upload", { method: "POST", body: fd });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || !data || !data.ok) {
    const msg = (data && data.error) ? data.error : `Upload failed (HTTP ${resp.status})`;
    throw new Error(msg);
  }
  return data.zip_path; // relative to input/
}

app.registerExtension({
  name: "comfyui.ziptools.upload_button",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ZipUploadButton") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      // zip_path is a STRING input; ComfyUI turns it into a text widget
      const pathWidget = findWidget(this, "zip_path");
      if (!pathWidget) return r;

      // Add Upload button widget
      const self = this;
      this.addWidget("button", "Upload .zip", "Choose file", async () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".zip,application/zip";
        input.onchange = async () => {
          if (!input.files || !input.files.length) return;
          const file = input.files[0];
          try {
            pathWidget.value = "Uploading...";
            self.setDirtyCanvas(true, true);
            const relPath = await uploadZip(file);
            pathWidget.value = relPath;
            self.setDirtyCanvas(true, true);
          } catch (e) {
            pathWidget.value = "";
            self.setDirtyCanvas(true, true);
            alert(e?.message || String(e));
          }
        };
        input.click();
      });

      return r;
    };
  },
});
