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

function getExistingPromptMaxIndex(node) {
  let max = 0;
  for (const inp of node.inputs || []) {
    if (!inp?.name) continue;
    const m = /^prompt_(\d+)$/.exec(inp.name);
    if (!m) continue;
    const n = Number(m[1]);
    if (Number.isFinite(n)) max = Math.max(max, n);
  }
  return max;
}

function setVisibleIfExists(node, idx, visible) {
  const input = findInput(node, `prompt_${idx}`);
  if (input) input.hidden = !visible;
}

function isLinked(node, idx) {
  const name = `prompt_${idx}`;
  const input = findInput(node, name);
  return Boolean(input && input.link != null);
}

function refreshVisibility(node) {
  // Only materialize ports on-demand.
  // Strategy:
  //  - Determine the last connected prompt slot that already exists.
  //  - Ensure the next slot exists (keep one empty slot).
  //  - Show only up to that next slot; hide any existing slots after it.

  let lastUsed = 0;
  const existingMax = getExistingPromptMaxIndex(node);
  for (let i = 1; i <= existingMax; i++) {
    if (isLinked(node, i)) lastUsed = i;
  }

  const showUpTo = Math.min(Math.max(lastUsed + 1, 1), MAX_PROMPTS);

  // Ensure ports up to showUpTo exist and are visible.
  for (let i = 1; i <= showUpTo; i++) {
    const input = ensureInputSlot(node, i);
    if (input) input.hidden = false;
  }

  // Hide any already-created ports beyond showUpTo.
  const newExistingMax = getExistingPromptMaxIndex(node);
  for (let i = showUpTo + 1; i <= newExistingMax; i++) {
    setVisibleIfExists(node, i, false);
  }

  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "jayhuang.promptlist.unlimited.autoports",
  nodeCreated(node) {
    if (node.comfyClass !== CLASS_NAME) return;

    // Materialize ONLY prompt_1 on creation.
    const first = ensureInputSlot(node, 1);
    if (first) first.hidden = false;

    // React to connecting/disconnecting.
    const orig = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
      if (orig) orig.apply(this, args);
      refreshVisibility(node);
    };

    refreshVisibility(node);
  },
});
