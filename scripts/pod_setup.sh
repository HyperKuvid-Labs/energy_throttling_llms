#!/usr/bin/env bash
# Provision a bare Ubuntu 22.04 GPU pod for the EAGLE3 sweep.
#
# The DataCrunch image ships no pip, no conda and no torch -- only the NVIDIA
# driver and Docker -- so python3-pip has to come from apt before anything else.
set -x

export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1

apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git
python3 -m pip install --upgrade pip -q
echo "=== PIP READY ==="
python3 -m pip --version

# sglang[all] pulls torch, sgl-kernel and flashinfer at versions matched to each
# other. Installing them separately is what produced the local ABI mismatch
# (sgl_kernel undefined symbol against torch 2.6).
python3 -m pip install -q "sglang[all]==0.5.2" 2>&1 | tail -5
python3 -m pip install -q pynvml psutil numpy requests huggingface_hub hf_transfer 2>&1 | tail -3

echo "=== VERSIONS ==="
python3 -c "import sglang; print('sglang', sglang.__version__)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python3 -c "import sgl_kernel; print('sgl_kernel OK')" || echo "sgl_kernel FAILED"
python3 -c "import flashinfer; print('flashinfer OK')" || echo "flashinfer MISSING"

echo "=== DOWNLOAD ==="
# Both repos are ungated, so no HF token is needed and none is copied here.
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["NousResearch/Meta-Llama-3.1-8B-Instruct",
             "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"]:
    p = snapshot_download(repo, ignore_patterns=["*.pth", "original/*"])
    print("downloaded", repo, "->", p, flush=True)
PY

echo "=== SETUP DONE ==="
