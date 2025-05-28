#!/usr/bin/env bash
# download_imagenet.sh ─────────────────────────────────────────────────────
# Resumable fetcher + extractor for ILSVRC‑2012 train/val archives.
# Two download back‑ends are supported:
#   1. Official ImageNet servers (requires IMAGENET_USERNAME / IMAGENET_ACCESS_KEY).
#   2. Kaggle competition mirror  (requires kaggle API credentials).
# -------------------------------------------------------------------------
# Usage examples
#   # Preferred (fastest – ImageNet servers)
#   IMAGENET_USERNAME=user IMAGENET_ACCESS_KEY=key \
#       ./download_imagenet.sh /data/imagenet
#
#   # Alternative (Kaggle mirror)
#   KAGGLE_USERNAME=user KAGGLE_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
#       ./download_imagenet.sh /data/imagenet
#
# The script is idempotent: it skips files already present and resumes
# partial downloads. Extracted structure:
#   imagenet/
#     ├── train/  (1000 class folders)
#     └── val/    (1000 class folders)
# -------------------------------------------------------------------------
set -euo pipefail
ROOT=${1:-$HOME/datasets/imagenet}
mkdir -p "$ROOT"
cd "$ROOT"

TRAIN_TAR=ILSVRC2012_img_train.tar
VAL_TAR=ILSVRC2012_img_val.tar

# ─────────────────────────  HELPERS  ──────────────────────────
join_by() { local IFS="$1"; shift; echo "$*"; }

retry_fetch() {
  local url=$1 file=$2
  if command -v aria2c &>/dev/null; then
    aria2c -x16 -s16 -k1M -o "$file" "$url" || true
  else
    wget -c --tries=3 --timeout=30 -O "$file" "$url"
  fi
}

kaggle_fetch() {
  local file=$1
  if ! command -v kaggle &>/dev/null; then
    echo "❌ kaggle CLI is not installed.  Install with: pip install kaggle" >&2; exit 1
  fi
  # ensure credentials are configured (either env or ~/.kaggle/kaggle.json)
  kaggle competitions download -c imagenet-object-localization-challenge -f "$file" -p . --quiet
}

extract_val() {
  echo "Extracting validation set …"
  mkdir -p val && tar xf "$VAL_TAR" -C val
  # organise into per‑class folders
  curl -sL https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -o val_map.sh
  awk '{print $2}' val_map.sh | sort -u | xargs -I{} mkdir -p val/{}
  awk '{print $1" " $2}' val_map.sh | while read img cls; do mv "val/$img" "val/$cls/"; done
  rm val_map.sh
}

extract_train() {
  echo "Extracting training set … this can take a while"
  mkdir -p train && tar xf "$TRAIN_TAR" -C train
  find train -name '*.tar' | parallel "cls={/%;s/.tar//}; mkdir -p train/$cls && tar xf {} -C train/$cls && rm {}"
}

# ────────────────────  DOWNLOAD SECTION  ─────────────────────
need_download=()
[[ -f $TRAIN_TAR ]] || need_download+=("$TRAIN_TAR")
[[ -f $VAL_TAR   ]] || need_download+=("$VAL_TAR")

if (( ${#need_download[@]} )); then
  echo "Files to fetch: $(join_by , ${need_download[@]})"
  # prefer ImageNet creds if supplied; else fallback to Kaggle
  if [[ -n "${IMAGENET_USERNAME:-}" && -n "${IMAGENET_ACCESS_KEY:-}" ]]; then
    echo "⏬ Using ImageNet servers …"
    for f in "${need_download[@]}"; do
      retry_fetch "https://${IMAGENET_USERNAME}:${IMAGENET_ACCESS_KEY}@image-net.org/data/ILSVRC/2012/$f" "$f"
    done
  else
    echo "⏬ Using Kaggle mirror …"
    for f in "${need_download[@]}"; do
      kaggle_fetch "$f"
    done
  fi
fi

# ────────────────────  EXTRACTION SECTION  ───────────────────
[[ -d val   ]] || extract_val
[[ -d train ]] || extract_train

echo "✅ ImageNet ready at $ROOT"
