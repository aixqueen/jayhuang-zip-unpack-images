import { app } from "/scripts/app.js";

function getPromptWidgets(node) {
  const ws = (node.widgets || []).filter(w => /^prompt_\d+$/.test(w.name));
  ws.sort((a, b) => {
    const ai = parseInt(a.name.split("_")[1] || "0", 10);
    const bi = parseInt(b.name.split("_")[1] || "0", 10);
    return ai - bi;
  });
  return ws;
}

function addPromptWidget(node, idx) {
  const name = `prompt_${idx}`;
  const w = node.addWidget(
    "text",
    name,
    "",
    () => {
      ensureTrailingEmpty(node);
    },
    { multiline: true }
  );
  // Keep a predictable ordering in the serialized workflow
  w.serializeValue = () => w.value;
  return w;
}

function ensureTrailingEmpty(node) {
  const ws = getPromptWidgets(node);
  if (ws.length === 0) return;

  // If the last widget has content, append a new empty prompt widget.
  const last = ws[ws.length - 1];
  const lastVal = (last.value ?? "").toString().trim();
  if (lastVal !== "") {
    addPromptWidget(node, ws.length + 1);
  }

  // Remove extra trailing empty widgets, keep exactly one empty at the end.
  const ws2 = getPromptWidgets(node);
  let i = ws2.length - 1;
  // Count trailing empties
  while (i >= 0 && (ws2[i].value ?? "").toString().trim() === "") {
    i--;
  }
  // i is last non-empty index; we want exactly one empty widget after it.
  const keepCount = Math.max(i + 2, 1);
  if (ws2.length > keepCount) {
    const remove = ws2.slice(keepCount);
    for (const w of remove) {
      const idx = node.widgets.indexOf(w);
      if (idx >= 0) node.widgets.splice(idx, 1);
    }
  }

  node.setSize(node.computeSize());
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "jayhuang.promptlist.unlimited.autogrow",
  nodeCreated(node) {
    if (node.comfyClass !== "easy promptList (Unlimited)") return;

    // Ensure we start with prompt_1 and one empty trailing slot.
    const ws = getPromptWidgets(node);
    if (ws.length === 0) {
      addPromptWidget(node, 1);
    }
    ensureTrailingEmpty(node);
  },
});
