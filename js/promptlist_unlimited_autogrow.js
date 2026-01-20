import { app } from "/scripts/app.js";

// Auto-growing INPUT PORTS for: easy promptList (Unlimited)
//
// Why this is needed:
//   In ComfyUI, optional inputs declared in Python are not automatically materialized
//   as LiteGraph input slots. We create the slots in JS and then show/hide them.

const MAX_PROMPTS = 64;
const CLASS_NAME = "easy promptList (Unlimited)";

function findInput(node, name) {
  return (node.inputs || []).find((i) => i && i.name === name);
}

function ensureInputSlot(node, idx) {
  const name = `prompt_${idx}`;
  let input = findInput(node, name);
  if (!input) {
    // Create LiteGraph input slot. Type STRING to match backend.
    node.addInput(name, "STRING");
    input = findInput(node, name);
  }
  return input;
}

function setVisible(node, idx, visible) {
  const input = ensureInputSlot(node, idx);
  if (input) input.hidden = !visible;
}

function isLinked(node, idx) {
  const name = `prompt_${idx}`;
  const input = findInput(node, name);
  return Boolean(input && input.link != null);
}

function refreshVisibility(node) {
  // Find last linked prompt
  let lastUsed = 0;
  for (let i = 1; i <= MAX_PROMPTS; i++) {
    if (isLinked(node, i)) lastUsed = i;
  }

  // Show up to lastUsed + 1 (keep one empty slot), at least 1
  const showUpTo = Math.min(Math.max(lastUsed + 1, 1), MAX_PROMPTS);

  for (let i = 1; i <= MAX_PROMPTS; i++) {
    setVisible(node, i, i <= showUpTo);
  }

  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "jayhuang.promptlist.unlimited.autoports",
  nodeCreated(node) {
    if (node.comfyClass !== CLASS_NAME) return;

    // Ensure slots exist, but hide everything except prompt_1.
    for (let i = 1; i <= MAX_PROMPTS; i++) {
      setVisible(node, i, i === 1);
    }

    // React to connecting/disconnecting.
    const orig = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      if (orig) orig.apply(this, args);
      refreshVisibility(node);
    };

    refreshVisibility(node);
  },
});
