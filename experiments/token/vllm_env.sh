# Source me before running anything that loads vllm.
#
# gpuc pins a job to its cards by UUID:
#
#   CUDA_VISIBLE_DEVICES=GPU-80646905-50a9-afc1-4375-43ca475b15e4
#
# which is the robust form - immune to device reordering between probe and run.
# torch accepts it, which is why every training and attribution job here works.
# vllm 0.11.0 does not: `Platform.device_id_to_physical_device_id` calls
# `int()` on the entry, so loading any model dies with
#
#   ValueError: invalid literal for int() with base 10: 'GPU-8064...'
#
# buried under a generic "Model architectures ['Qwen2ForCausalLM'] failed to be
# inspected", because the real error happens in a subprocess.
#
# Rewrite the UUIDs to the physical indices they name. Same cards, integer form.
# nvidia-smi is queried with the variable unset, since it honours it too and
# would otherwise only report the cards we are pinned to.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(python3 - <<'PYEOF'
import os, subprocess, sys

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
entries = [e.strip() for e in visible.split(",") if e.strip()]
if not any(e.startswith("GPU-") for e in entries):
    print(visible)
    sys.exit()

environment = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
listing = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
    capture_output=True, text=True, check=True, env=environment,
).stdout
index_of = {}
for line in listing.strip().splitlines():
    index, uuid = (part.strip() for part in line.split(","))
    index_of[uuid] = index

missing = [e for e in entries if e.startswith("GPU-") and e not in index_of]
if missing:
    sys.exit(f"CUDA_VISIBLE_DEVICES names unknown GPU UUIDs: {missing}")
print(",".join(index_of.get(e, e) for e in entries))
PYEOF
)
  export CUDA_VISIBLE_DEVICES
  echo "CUDA_VISIBLE_DEVICES rewritten for vllm: $CUDA_VISIBLE_DEVICES"
fi
