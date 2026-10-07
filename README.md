# VeriTrace-RAG

**Trustworthy retrieval-augmented generation with a query-adaptive hybrid retriever built on our own
inverted index, and claim-level citation verification.**
CSD358 Information Retrieval, mid-semester hackathon, **Track T1 (RAG and trustworthy answers)**.
Team: Koneru Akhil (2410110564), Velagala Surya Prakash Reddy (2410110376).

An LLM is only as good as what gets retrieved for it, and a citation is only useful if it really supports
the sentence it is attached to. VeriTrace-RAG answers scientific questions (or checks scientific claims) over
the SciFact corpus. Every sentence of the answer is traced to a ranked source *sentence*, wrong citations are
repaired, unsupported or contradicted claims are flagged, and the system says so when retrieval itself failed.

## What is new (vs. the obvious baseline and existing RAG toolkits)

| | Idea | Where |
|---|---|---|
| **N1** | **Query-adaptive hybrid retrieval.** A learned router predicts, per query, whether lexical (BM25 over our inverted index) or dense evidence should dominate, using query-performance-prediction (QPP) features computed from index statistics (idf, SCQ, score shape, top-10 agreement) plus a cheap cross-encoder arbitration, and sets the fusion weight a(q) accordingly. | `veritrace/router.py`, `veritrace/qpp.py` |
| **N2** | **Claim-level provenance audit with repair.** Each generated sentence is a claim; a trained verifier built mostly on IR features (lnc.ltc cosine, idf-weighted coverage, the idf of the most informative missing term, biword coverage, numeric consistency, polarity) plus NLI decides VERIFIED / REPAIRED (a different retrieved source supports it) / NEW SOURCE (the claim is re-issued as a query against the whole index: *verify-by-retrieval*) / CONTRADICTED / UNSUPPORTED, and points to the exact evidence sentence. | `veritrace/verify.py` |
| **N3** | **idf-guided hallucination injection** for training and stress-testing the verifier without labels: negation, direction flip, number swap, and an entity swap that replaces the claim's highest-idf term by a term of similar idf. | `veritrace/corrupt.py` |
| **N4** | **Knowing when not to answer.** An out-of-scope gate returns **zero documents** when even the best candidate is judged irrelevant (calibrated on training questions); the same QPP signals estimate P(a relevant source is in the top-5), and low confidence leads to an abstention or warning, evaluated with risk-coverage curves. | `veritrace/pipeline.py`, `veritrace/abstain.py`, `scripts/calibrate_scope_gate.py` |

## Key results (SciFact test, 300 expert-judged claims)

| System | nDCG@10 | P@1 | R@10 |
|---|---|---|---|
| tf-idf lnc.ltc (lecture baseline) | 0.647 | 0.510 | 0.788 |
| BM25 + learned title/body zones | 0.680 | 0.557 | 0.820 |
| Dense BGE-small (zero-shot) / fine-tuned with BM25-mined negatives | 0.712 / 0.745 | 0.603 / 0.620 | 0.831 / 0.878 |
| Hybrid, best fixed fusion weight | 0.735 | 0.627 | 0.851 |
| **Adaptive router (N1)** | **0.747** (p = 0.02) | 0.630 | 0.870 |
| **Router + LambdaMART** | **0.773** (p = 0.002) | **0.667** | 0.877 |

| Claim checker (908 human-labelled dev pairs) | AUROC | 3-way macro-F1 | negations caught (paired) |
|---|---|---|---|
| cosine(claim, cited chunk), the track's sample idea | 0.775 | - | 0.56 (chance) |
| NLI only | 0.870 | 0.574 | 0.72 |
| **VeriTrace verifier (N2 + N3)** | **0.902** | **0.666** | **0.85** |

End to end on 30 held-out test claims (Qwen2.5-1.5B on CPU, independent DeBERTa-v3-large judge): the LLM's own
citations support only 46% of its claims; 88% of the claims VeriTrace keeps are confirmed by the judge; hallucinations
injected into real answers are flagged 71% of the time vs. 4% by the cosine checker.

Off-topic questions (N4): zero documents for 39 of 40 general-knowledge questions and 97.5% of FiQA
finance questions, while only 2.7% of SciFact test claims are rejected.
Retrieval-failure prediction (N4): AUROC 0.802 vs. 0.731 for the best single score. Self-judged questions:
P@1 0.90 vs. 0.50 for tf-idf. Router also significant on NFCorpus (0.373 vs. 0.366, p = 0.014); on FiQA, where
dense retrieval dominates, it stays at the dense-heavy fixed weight (no significant change), while RRF loses 5 points.

IR design experiments: Porter stemming 0.680 vs. 0.651 nDCG@10 without stemming; idf is the largest gain among
SMART schemes (lnc.lnc 0.534 → lnc.ltc 0.647); a tiered index (r = 100) keeps 0.681 while scoring 17% of the chunks; query
spelling correction recovers nDCG@10 from 0.614 to 0.669 under two typos per claim.

Warm-model latency on a 16-core laptop CPU: retrieval 0.8 s, Qwen2.5-1.5B generation ~42 s, claim audit ~14 s.

## IR principles used (and where)

| Principle (lectures) | Code |
|---|---|
| What is a document: sentence-window chunking vs. whole abstract, evaluated | `veritrace/chunking.py`, `scripts/exp_chunking.py` |
| Tokenisation, case folding, stop words, Porter stemming | `veritrace/text.py` |
| Inverted index: dictionary (df, cf) + positional postings (tf, positions) | `veritrace/index.py` |
| Zone index (title / body) and **learned zone weights** | `veritrace/index.py`, `veritrace/sparse.py`, `scripts/exp_retrieval.py` |
| Boolean AND / OR / NOT, phrase queries (positional intersect), skip pointers, df-ordered conjunctions | `veritrace/boolean.py` |
| tf-idf, SMART **lnc.ltc** cosine, length normalisation | `veritrace/sparse.py`, `veritrace/verify.py` |
| BM25 (beyond syllabus) | `veritrace/sparse.py` |
| Heap-based top-K, **champion lists** (impact-ordered postings), index elimination | `veritrace/sparse.py`, `scripts/exp_efficiency.py` |
| Query-term proximity and phrase features | `veritrace/sparse.py`, `veritrace/ltr.py` |
| Dense retrieval and learning to rank (LambdaMART), beyond syllabus | `veritrace/dense.py`, `veritrace/ltr.py` |
| Precision / recall / P@k / MAP / MRR / nDCG, pooling for self-judged queries | `veritrace/metrics.py`, `scripts/judge_pool.py` |
| **Heap top-K on the live path**: the candidate pool of every query is selected with a size-n min-heap | `veritrace/fusion.py` (`heap_select`) |
| **Tiered index** (tier 1 = champion lists, tier 2 = full postings) and **many-term matching** index elimination | `veritrace/sparse.py` (`search_tiered`, `bm25_min_match`), `scripts/exp_efficiency.py` |
| **SMART weighting schemes** (nnn ... lnc.ltc ... ltn.ltn) and the **Jaccard** coefficient, compared | `veritrace/sparse.py` (`smart_zone`, `jaccard_zone`), `scripts/exp_ir_extras.py smart` |
| **Analysis-chain ablation**: Porter vs. S-stemmer vs. no stemming, stop list vs. none | `veritrace/text.py` (`Analyzer`), `scripts/exp_ir_extras.py analysis` |
| **Spelling correction**: bigram k-gram index + Jaccard, Damerau edit distance, Soundex fallback, postings-intersection context check | `veritrace/spell.py`, `scripts/exp_ir_extras.py spell` |
| **Cluster pruning** (leaders / followers) for the dense index | `veritrace/cluster.py`, `scripts/exp_ir_extras.py cluster` |
| Per-term score breakdown (df, idf, tf, BM25 and lnc.ltc contribution of each term) | `veritrace/sparse.py` (`explain`), `scripts/worked_example.py` |

## Quick start: the minimum to run the demo

If you only want the live system (not every experiment), these are the steps it needs, in order:

```bash
pip install -r requirements.txt
python scripts/download.py                       # data + models
python scripts/build_index.py --dataset scifact  # inverted index + dense vectors
python scripts/exp_retrieval.py --dataset scifact   # BM25/zone tuning, router (writes artifacts/scifact/router.pkl)
python scripts/exp_ltr.py --dataset scifact      # LambdaMART (ltr.pkl)
python scripts/train_abstention.py --dataset scifact # retrieval confidence (abstain.pkl)
python scripts/calibrate_scope_gate.py           # out-of-scope gate (scope_gate.json)
python scripts/train_verifier.py                 # claim verifier (verifier.pkl)
streamlit run app.py                             # or: python cli.py "your question" --generator extractive
```

## Setup

Tested on Windows 11, Python 3.13 (Anaconda), CPU only (16-core laptop, no GPU needed).

```bash
pip install -r requirements.txt
python scripts/download.py          # datasets (~30 MB) + pretrained models (~5 GB incl. Qwen2.5-1.5B)
```

On Windows with Anaconda, torch and LightGBM both bundle OpenMP; the package sets
`KMP_DUPLICATE_LIB_OK=TRUE` automatically when `veritrace` is imported.

## Reproduce everything

```bash
python scripts/build_index.py --dataset scifact                 # chunks + inverted index + dense vectors
python scripts/build_index.py --dataset scifact --policy doc    # whole-abstract baseline index
python scripts/exp_retrieval.py --dataset scifact               # BM25/zones tuning, pools, router, baselines
python scripts/exp_ltr.py --dataset scifact                     # LambdaMART + ablations
python scripts/train_abstention.py --dataset scifact            # retrieval-confidence model (N4)
python scripts/calibrate_scope_gate.py                          # out-of-scope gate: when to return 0 documents (N4)
python scripts/train_verifier.py                                # claim verifier (N2 + N3)
python scripts/exp_efficiency.py                                # champion lists / index elimination
python scripts/exp_chunking.py                                  # what is a document?
python scripts/train_dense.py --dataset scifact                 # optional: dense fine-tuning with IR-mined negatives
python scripts/eval_rag.py --generator qwen --n 50 --tag qwen_v1_raw     # end-to-end generation + audit
python scripts/eval_rag.py --reaudit results/rag_eval_qwen_v1_raw.json --tag qwen_reaudit
python scripts/eval_rag.py --generator qwen --n 30 --seed 99 --exclude results/rag_eval_qwen_v1_raw.json --tag qwen_fresh
python scripts/eval_rag.py --generator extractive --n 30 --seed 7 --tag extractive
python scripts/verifier_pairwise.py                             # threshold-free injected-hallucination test
python scripts/measure_latency.py --generator qwen --n 5        # warm-model latency
python scripts/judge_pool.py --make / --eval                    # self-judged natural-language queries
python scripts/exp_ir_extras.py smart analysis spell cluster    # SMART schemes, analysis chain, spelling, cluster pruning
                                                                # (cluster uses the fine-tuned encoder: run train_dense.py first)
python scripts/worked_example.py                                # every intermediate number for one query
python scripts/make_figures.py                                  # all figures -> results/figures
```

Note: the first `eval_rag.py` run above was made with the sentence-level audit (v1); the claim normalisation
of audit v2 was added afterwards, so re-running it today gives v2 verdicts. The saved v1 results are in
`results/rag_eval_qwen_v1_raw.json`.

NFCorpus and FiQA (generalisation of the router and LambdaMART) use the same three first commands
with `--dataset nfcorpus` / `--dataset fiqa`.

## Run the system

```bash
streamlit run app.py
```

```bash
python cli.py "Does vitamin D deficiency increase the risk of multiple sclerosis?" --generator qwen
```

```bash
python cli.py --term methadone --boolean "\"liver transplantation\" AND methadone NOT children"
```
(In Windows PowerShell, add `--%` before `--boolean` so the inner quotes reach Python unchanged:
`python cli.py --term methadone --% --boolean "\"liver transplantation\" AND methadone NOT children"`.)

The CLI prints every intermediate artefact: the analysis chain, df/idf of each query term, a term's
postings with positions and its champion list, BM25 and dense top lists, the QPP features and the router's
fusion weight, the LambdaMART feature table, the retrieval confidence, the answer, and the claim audit.
`--audit-text "..."` audits any answer you type (try changing a number in a correct answer).
Misspelled words are always reported as a "did you mean" suggestion; `--spell` searches with the corrected
query (try `"Does vitamin D deficency increse the risk of multiple sclerosis?" --spell`). The CLI also prints
the per-term BM25 / lnc.ltc breakdown of the top chunk and whether the tiered index answered from tier 1.
Generators: `extractive` (no LLM, fast), `qwen0.5b`, `qwen` (Qwen2.5-1.5B-Instruct, ~40 s on CPU),
`openai` (any OpenAI-compatible endpoint via `VT_LLM_BASE_URL`, `VT_LLM_MODEL`, `VT_LLM_API_KEY`).

## Data

* **SciFact** (Wadden et al., 2020): 5,183 abstracts, 809 train / 300 test claims, sentence-level rationales. CC BY-NC 2.0.
* **NFCorpus** (Boteva et al., 2016) and **FiQA-2018** (Maia et al., 2018), BEIR versions (Thakur et al., 2021).
* Pretrained models from the Hugging Face hub: BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2,
  MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (verifier NLI feature),
  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (independent judge, evaluation only),
  Qwen/Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct (generation).

No crawling was performed; no personal data is collected.

## What works and what is planned

Works: everything above runs end to end on CPU; every number in this README is produced by the scripts
in `scripts/` and stored in `results/` (figures in `results/figures/`).

Limitations and next steps: claims are checked as whole sentences rather than atomic facts, and the audit
checks attribution, not relevance to the question. Planned: claim decomposition into atomic facts,
fine-tuning the verifier's NLI component on SciFact, a relevance check, and a larger corpus with
compressed postings and an approximate-nearest-neighbour dense index.

## Repository layout

```
veritrace/          the IR system (one module per stage: text, index, sparse, dense, router, verify, ...)
scripts/            experiments, training and evaluation (each writes results/*.json)
app.py, cli.py      demo front-ends
data/               raw datasets (downloaded) + our self-judged queries
artifacts/          indexes, embeddings, trained models (generated)
results/            metrics (json) and figures
```
