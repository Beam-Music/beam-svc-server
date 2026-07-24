#!/usr/bin/env bash
set -euo pipefail

# Public MedleyVox reproduction, CC BY 4.0:
# https://huggingface.co/Cyru5/MedleyVox
# The original MedleyVox repository does not publish checkpoints.

output_dir="${1:-weights/medleyvox}/checkpoint/singing_librispeech_ft_iSRNet"
mkdir -p "$output_dir"

curl -fL --retry 3 \
  "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.json" \
  -o "$output_dir/vocals.json"
curl -fL --retry 3 \
  "https://huggingface.co/Cyru5/MedleyVox/resolve/main/singing_librispeech_ft_iSRNet/vocals.pth" \
  -o "$output_dir/vocals.pth"

shasum -a 256 "$output_dir/vocals.json" "$output_dir/vocals.pth"
