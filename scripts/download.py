import argparse
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from veritrace.config import (CROSS_ENCODER, DENSE_MODEL, JUDGE_NLI_MODEL, LLM_MODEL, LLM_MODEL_SMALL,
                              NLI_MODEL, RAW)

# beir copies of scifact, nfcorpus and fiqa + the original scifact release with rationales
BEIR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
SCIFACT = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def fetch(url, dest):
    if dest.exists():
        print("exists", dest)
        return
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--datasets", default="scifact,nfcorpus,fiqa")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets.split(","):
        z = RAW / f"{ds}.zip"
        fetch(BEIR.format(ds), z)
        if not (RAW / ds).exists():
            zipfile.ZipFile(z).extractall(RAW)
    t = RAW / "scifact_orig.tar.gz"
    fetch(SCIFACT, t)
    if not (RAW / "scifact_orig" / "data").exists():
        tarfile.open(t).extractall(RAW / "scifact_orig")
    if not args.no_models:
        from huggingface_hub import snapshot_download
        for m in (DENSE_MODEL, CROSS_ENCODER, NLI_MODEL, LLM_MODEL, LLM_MODEL_SMALL, JUDGE_NLI_MODEL):
            print("model", m)
            snapshot_download(m, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*", "merges.txt", "vocab*"])


if __name__ == "__main__":
    main()
