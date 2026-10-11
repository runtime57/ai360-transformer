#!/usr/bin/env bash
# Push code to the pod; record repo HEAD + content hash of src/scripts/configs as the code version.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
POD="${POD:-root@195.26.233.28}"; PORT="${PORT:-43356}"
HEAD=$(git -C "$HERE" rev-parse --short HEAD)
CODE=$(cd "$HERE" && find src scripts configs -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) | sort | xargs cat | sha256sum | cut -c1-12)
echo "${HEAD}+code.${CODE}" > "$HERE/CODE_VERSION"
rsync -rltz --no-owner --no-group -e "ssh -p $PORT" --exclude results --exclude work --exclude '__pycache__' "$HERE/" "$POD:/workspace/qlens/"
echo "synced ${HEAD}+code.${CODE}"
