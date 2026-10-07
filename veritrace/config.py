from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

for _p in (ARTIFACTS, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# all pretrained models are downloaded from the hugging face hub
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
JUDGE_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"

DATASETS = ("scifact", "nfcorpus", "fiqa")


def art(dataset: str) -> Path:
    p = ARTIFACTS / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p
