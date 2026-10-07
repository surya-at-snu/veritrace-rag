import argparse
import sys
import textwrap
import time

import veritrace

W = 100
STATUS_MARK = {"VERIFIED": "[OK] VERIFIED", "REPAIRED": "[FIX] REPAIRED", "REPAIRED_NEW": "[NEW] REPAIRED+NEW SOURCE",
               "CONTRADICTED": "[X] CONTRADICTED", "UNSUPPORTED": "[!] UNSUPPORTED",
               "NOT_A_CLAIM": "[-] meta statement, not verified"}


def hr(title=""):
    print("\n" + ("== " + title + " ").ljust(W, "="))


def wrap(s, indent=4):
    return textwrap.fill(s, W, initial_indent=" " * indent, subsequent_indent=" " * indent)


def show_term(vt, term):
    from veritrace.text import stem
    t = stem(term.lower())
    e = vt.index.describe_term(t)
    hr(f"Dictionary entry for '{term}' -> index term '{t}'")
    if not e["df"]:
        print("    not in the dictionary")
        return
    print(f"    df={e['df']} chunks   cf={e['cf']}   idf=log10(N/df)={e['idf']}   title-zone df={e['title_df']}")
    print("    postings (chunk id: tf @ positions):")
    for p in e["postings"]:
        print(f"      chunk {p['chunk']:>6} (doc {p['doc']}): tf={p['tf']} @ {p['positions']}")
    print(f"    champion list (top-5 by BM25 impact): {e['champions']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--generator", default="extractive", choices=["extractive", "qwen", "qwen0.5b", "openai", "none"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--filter", default=None, help="Boolean pre-filter, e.g. '\"vitamin d\" AND sclerosis NOT mice'")
    ap.add_argument("--strict", action="store_true", help="abstain when retrieval confidence is low")
    ap.add_argument("--term", default=None, help="show the dictionary entry and postings of a term")
    ap.add_argument("--boolean", default=None, help="run a pure Boolean query")
    ap.add_argument("--spell", action="store_true", help="correct misspelled query words (k-gram index + edit "
                    "distance) before retrieval; without it the correction is only suggested")
    ap.add_argument("--audit-text", default=None, help="audit this answer text instead of generating one "
                    "(e.g. paste an answer with a deliberately wrong number)")
    args = ap.parse_args()

    from veritrace.pipeline import VeriTrace, explain_query_processing
    t0 = time.time()
    vt = VeriTrace(args.dataset, generator=None if args.generator == "none" else args.generator)
    print(f"[loaded index: {vt.index.n_docs} docs, {vt.index.n_chunks} chunks, body vocabulary "
          f"{len(vt.index.body.vocab)} terms, {int(vt.index.body.df.sum())} postings in {time.time() - t0:.1f}s]")

    if args.term:
        show_term(vt, args.term)
    if args.boolean:
        hr(f"Boolean query: {args.boolean}")
        res = vt.boolean.query(args.boolean)
        print(f"    parse tree: {vt.boolean.parse(args.boolean)}")
        print(f"    {len(res)} matching chunks")
        for c in res[:10]:
            ch = vt.index.chunks[c]
            print(f"      chunk {c:>6}  doc {ch.doc_id}: {ch.title[:80]}")
    if not args.query:
        return

    q = args.query
    corrected, fixes = vt.suggest(q)
    if fixes:
        hr("0. Spelling correction (bigram k-gram index -> Jaccard >= 0.35 -> edit distance -> context check)")
        for f in fixes:
            print(f"    {f.original!r} is not in the vocabulary -> {f.corrected!r}  ({f.method}, edit distance "
                  f"{f.distance}, bigram Jaccard {f.jaccard})")
        print(f"    did you mean: {corrected}")
        if args.spell:
            q = corrected
            print("    (--spell: using the corrected query)")
        else:
            print("    (run with --spell to search with the corrected query)")
    hr("1. Query processing (case folding -> tokens -> stop words -> Porter stems)")
    print("    " + "  ".join(f"{raw}->{t}" for raw, t in explain_query_processing(q)))
    # run retrieval once and print every intermediate step
    r = vt.retrieve(q, args.k, args.filter)
    print("    term statistics (df over chunks, idf = log10(N/df), title-zone df):")
    for t, df, idf, tdf in sorted(r.term_stats, key=lambda x: -x[2]):
        print(f"      {t:16s} df={df:<6} idf={idf:<6} title_df={tdf}")
    if r.term_stats:
        top_term = max(r.term_stats, key=lambda x: x[2] if x[1] else -1)[0]
        show_term(vt, top_term)
    if args.filter:
        print(f"\n    Boolean filter {args.filter!r}: {r.n_filtered} chunks pass")

    hr("2. First-stage rankings")
    print("    BM25 (k1={}, b={}, title-zone weight g={})".format(vt.sparse.k1, vt.sparse.b, vt.sparse.g).ljust(52) + "Dense cosine (BGE-small)")
    for (d1, s1), (d2, s2) in zip(r.sparse_top[:5], r.dense_top[:5]):
        print(f"      {d1:>10} {s1:8.3f}".ljust(52) + f"  {d2:>10} {s2:6.3f}")

    # where does the BM25 score of the best chunk come from? one row per query term
    import numpy as np
    full = vt.sparse.chunk_scores(q)
    if full.max() > 0:
        top_chunk = int(np.argmax(full))
        rows = vt.sparse.explain(q, top_chunk)
        print(f"\n    BM25 score breakdown for the top chunk {top_chunk} (doc {vt.index.chunks[top_chunk].doc_id}):")
        print(f"      {'term':14s} {'df':>5} {'idf':>6} {'tf body':>7} {'BM25 body':>9} {'tf title':>8} "
              f"{'BM25 title':>10} {'(1-g)b+g*t':>10} {'lnc.ltc':>8}")
        for x in rows:
            print(f"      {x['term']:14s} {x['df']:>5} {x['bm25_idf']:>6.2f} {x['tf_body']:>7} {x['bm25_body']:>9.3f} "
                  f"{x['tf_title']:>8} {x['bm25_title']:>10.3f} {x['weighted']:>10.3f} {x['lnc_ltc']:>8.3f}")
        print(f"      {'total':14s} {'':>5} {'':>6} {'':>7} {sum(x['bm25_body'] for x in rows):>9.3f} {'':>8} "
              f"{sum(x['bm25_title'] for x in rows):>10.3f} {sum(x['weighted'] for x in rows):>10.3f} "
              f"{sum(x['lnc_ltc'] for x in rows):>8.3f}  (= score used for ranking: {full[top_chunk]:.3f})")
        # tiered index: does tier 1 (champion lists) already give the same top 10?
        hits_t, tier, n_t = vt.sparse.search_tiered(q, r=100, k=10)
        from veritrace.sparse import heap_topk
        exact = [c for c, _ in heap_topk(full, 10)]
        same = len(set(exact) & {c for c, _ in hits_t})
        print(f"\n    tiered index (tier 1 = champion lists, r=100): answered from tier {tier}, scored {n_t} chunks instead "
              f"of {int((full > 0).sum())}; {same}/10 of the exhaustive top-10 chunks found")

    hr("3. Query-adaptive fusion router (QPP + cross-encoder arbitration)")
    keys = ["q_len", "idf_avg", "idf_max", "idf_std", "scq_avg", "sp_top1", "sp_gap12", "de_top1", "de_gap12", "agree_jacc10"]
    print("    " + "  ".join(f"{k}={r.qpp[k]:.3f}" for k in keys))
    print("    " + "  ".join(f"{k}={v:.2f}" for k, v in r.ce_arb.items()))
    print(f"    P(lexical retriever wins) = {r.p_lexical:.3f}  ->  fusion weight a(q) on BM25 = {r.alpha:.2f} "
          f"(best fixed weight = {vt.alpha_fixed})")

    # nothing relevant in the corpus, stop here
    if r.out_of_scope:
        hr("OUT OF SCOPE -> 0 documents")
        print(f"    best cross-encoder relevance {r.best_ce:.2f} < threshold {r.scope_tau:.2f} (1st percentile of in-domain "
              f"training questions)")
        print("    the corpus has nothing about this question, so no documents are returned and nothing is generated.")
        return
    hr("4. LambdaMART re-ranking (top documents, MaxP over chunks)")
    print(f"    {'#':>2} {'doc':>10} {'LTR':>7} {'fused':>6} {'BM25':>7} {'dense':>6} {'CE':>6}  title")
    for i, x in enumerate(r.ranked[:8], 1):
        ltr = f"{x['ltr']:7.3f}" if x["ltr"] is not None else "     - "
        bm = f"{x['bm25']:7.2f}" if x["bm25"] is not None else "     - "
        de = f"{x['dense']:6.3f}" if x["dense"] is not None else "    - "
        ce = f"{x['ce']:6.2f}" if x["ce"] is not None else "    - "
        print(f"    {i:>2} {x['doc_id']:>10} {ltr} {x['fused']:6.3f} {bm} {de} {ce}  {x['title'][:46]}")
    tau = vt.conf.tau if vt.conf else 0
    print(f"\n    retrieval confidence P(relevant source in top-{args.k}) = {r.confidence:.3f} "
          f"(abstain threshold {tau:.3f}) -> {'LOW: answer withheld/flagged' if r.confidence < tau else 'OK'}")
    print("    timings: " + "  ".join(f"{k}={v:.0f}" for k, v in r.timings.items()))

    if vt.generator is None and not args.audit_text:
        return
    hr(f"5. Answer ({'user-supplied text' if args.audit_text else vt.generator.name}) with numbered sources")
    for i, ev in enumerate(r.contexts, 1):
        print(f"    [{i}] {ev.title[:90]}  (chunk {ev.ref})")
    a = vt.audit_text(r, args.audit_text) if args.audit_text else vt.answer(q, args.k, args.filter, strict=args.strict)
    print()
    print(wrap(a.raw_answer))

    hr("6. Claim-level citation audit")
    for i, v in enumerate(a.verdicts, 1):
        print(f"  claim {i}: {v.text}")
        src = f"source [{v.support_ref}]" if v.support_ref else (f"NEW chunk {v.new_chunk}" if v.new_chunk is not None else "-")
        print(f"     cited {v.cited} -> {STATUS_MARK[v.status]}  ({src}, P(support)={v.p_support:.2f}, P(contradict)={v.p_contradict:.2f})")
        if v.support_sentence:
            print(wrap("evidence: \"" + v.support_sentence[:300] + "\"", 9))
    print(f"\n    TRUST SCORE = {a.trust:.0%} of claims grounded"
          + ("   |   WARNING: low retrieval confidence" if a.abstained else ""))
    print("    timings: " + "  ".join(f"{k}={v:.1f}s" for k, v in a.timings.items()))


if __name__ == "__main__":
    sys.exit(main())
