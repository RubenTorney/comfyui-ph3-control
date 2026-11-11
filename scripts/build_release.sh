#!/usr/bin/env bash
# Builds dist/comfyui-ph3-control/ from the files tracked by git (local only, publishes nothing).
#   scripts/build_release.sh              full pack, including the small adaLN table
#   scripts/build_release.sh --no-table   without ph3_control/data/ph3_adaln_map.safetensors
# A --no-table pack still works if the MiniMax H3 fl2va file is in models/diffusion_models (the map is then computed
# locally), or after: python tools/build_adaln_table.py P-H3-IMAGE.safetensors H3_FL2VA.safetensors
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist/comfyui-ph3-control
rm -rf dist && mkdir -p "$OUT"
git ls-files -z | xargs -0 -I{} cp --parents {} "$OUT"
if [ "${1:-}" = "--no-table" ]; then
  rm -f "$OUT/ph3_control/data/ph3_adaln_map.safetensors"
  printf '
This build does not include the adaLN table mentioned above.
' >> "$OUT/NOTICE"
  echo "built WITHOUT the adaLN table"
fi
(cd dist && tar czf comfyui-ph3-control.tar.gz comfyui-ph3-control)
echo "$OUT ($(du -sh "$OUT" | cut -f1)), archive dist/comfyui-ph3-control.tar.gz"
