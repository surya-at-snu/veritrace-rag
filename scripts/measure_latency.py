import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from veritrace.config import RESULTS
from veritrace.corpus import split_queries
from veritrace.pipeline import VeriTrace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="qwen")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    vt = VeriTrace("scifact", generator=args.generator)
    load_s = time.time() - t0
    queries, _ = split_queries("scifact", "test")
    qs = [queries[q] for q in sorted(queries)[: args.n + 1]]
    # warm up run so lazy model loading isnt counted
    vt.answer(qs[0])
    rows = []
    for q in qs[1:]:
        a = vt.answer(q)
        rows.append({**a.retrieval.timings, **a.timings, "n_claims": len(a.verdicts)})
    out = {"generator": args.generator, "load_s": load_s, "n": len(rows),
           "mean": {k: float(np.mean([r.get(k, 0) for r in rows])) for k in rows[0]}}
    print(json.dumps(out, indent=1))
    json.dump(out, open(RESULTS / f"latency_{args.generator}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
