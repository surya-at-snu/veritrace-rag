import html
import json
import re
from pathlib import Path

import veritrace

import pandas as pd
import streamlit as st

from veritrace.config import FIGURES, RESULTS
from veritrace.text import analyze, stem

st.set_page_config(page_title="VeriTrace-RAG", layout="wide")

STATUS_STYLE = {
    "VERIFIED": ("#1a7f37", "verified by the cited source"),
    "REPAIRED": ("#0969da", "citation repaired: supported by a different retrieved source"),
    "REPAIRED_NEW": ("#8250df", "supported by a new source found by verify-by-retrieval"),
    "CONTRADICTED": ("#cf222e", "contradicted by the evidence"),
    "UNSUPPORTED": ("#bc4c00", "no supporting evidence found (possible hallucination)"),
    "NOT_A_CLAIM": ("#8a8984", "statement about the sources, not a factual claim (not verified)"),
}
EXAMPLES = [
    "Does vitamin D deficiency increase the risk of multiple sclerosis?",
    "32% of liver transplantation programs required patients to discontinue methadone treatment in 2001.",
    "What is the role of microRNAs in cancer metastasis?",
    "Statins reduce the risk of venous thromboembolism.",
    "Who won the 2018 FIFA World Cup?",
]


# load everything once and keep it in memory between reruns
@st.cache_resource(show_spinner="Loading index, models and trained components ...")
def load_system(generator: str):
    from veritrace.pipeline import VeriTrace
    return VeriTrace("scifact", generator=None if generator == "none" else generator)


def highlight(text: str, terms: set, sentence: str = "") -> str:
    out = html.escape(text)
    if sentence:
        s = html.escape(sentence)
        out = out.replace(s, f"<span style='background:rgba(46,160,67,.18);border-radius:3px'>{s}</span>")
    def bold(m):
        w = m.group(0)
        return f"<b>{w}</b>" if stem(w.lower()) in terms else w
    return re.sub(r"[A-Za-z][A-Za-z0-9\-]*", bold, out)


with st.sidebar:
    st.header("Settings")
    gen = st.selectbox("Generator", ["extractive", "qwen0.5b", "qwen", "none"],
                       help="qwen = Qwen2.5-1.5B-Instruct on CPU (slower, ~30 s); extractive = no LLM")
    k = st.slider("Sources given to the generator (k)", 3, 8, 5)
    bfilter = st.text_input("Boolean filter (optional)", placeholder='"vitamin d" AND sclerosis NOT mice')
    strict = st.checkbox("Abstain when retrieval confidence is low", value=False)
    spell = st.checkbox("Correct misspelled query words", value=True,
                        help="k-gram (bigram) index + Jaccard for candidates, edit distance to rank them, "
                             "co-occurrence check against the other query terms; only words not in the vocabulary")

st.title("VeriTrace-RAG")
st.caption("Query-adaptive hybrid retrieval over an inverted index + claim-level citation verification. "
           "Corpus: SciFact (5,183 scientific abstracts, 9,022 chunks).")

tab_ask, tab_ir, tab_index, tab_eval = st.tabs(["Ask", "Retrieval internals", "Index explorer", "Evaluation"])

with tab_ask:
    ex = st.selectbox("Example", ["(type your own)"] + EXAMPLES)
    q = st.text_input("Question or claim", value="" if ex == "(type your own)" else ex)
    go = st.button("Search & answer", type="primary")
    if go and q.strip():
        vt = load_system(gen)
        try:
            with st.spinner("Retrieving, generating and auditing ..."):
                ans = vt.answer(q, k, bfilter or None, strict=strict, spell=spell)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.session_state["ans"] = ans
    ans = st.session_state.get("ans")
    if ans and ans.retrieval.corrections:
        fx = ", ".join(f"{c['original']} → {c['corrected']} (edit distance {c['distance']})"
                       for c in ans.retrieval.corrections)
        st.info(f"Searched for: **{ans.retrieval.query}**  (spelling corrected: {fx})")
    # off topic question: show zero documents
    if ans and ans.retrieval.out_of_scope:
        r = ans.retrieval
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents returned", "0")
        c2.metric("Best relevance score", f"{r.best_ce:.1f}", f"threshold {r.scope_tau:.1f}", delta_color="off",
                  help="cross-encoder score of the best candidate chunk; in-scope science questions score about +3")
        c3.metric("Retrieval time", f"{r.timings['retrieval_total_ms'] / 1000:.1f} s")
        st.error("0 documents found. This question is outside the scope of the corpus (5,183 scientific abstracts): "
                 "even the best-matching chunk is judged irrelevant, so no answer is generated.")
        st.caption("The 'Retrieval internals' tab shows the closest (irrelevant) matches and why they were rejected.")
    elif ans:
        r = ans.retrieval
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trust score", f"{ans.trust:.0%}" if ans.verdicts else "-")
        c2.metric("Retrieval confidence", f"{r.confidence:.2f}", "low" if ans.abstained else "ok",
                  delta_color="inverse" if ans.abstained else "normal")
        c3.metric("BM25 weight a(q)", f"{r.alpha:.2f}", help="query-adaptive fusion weight from the router (N1)")
        c4.metric("Retrieval time", f"{r.timings['retrieval_total_ms'] / 1000:.1f} s", help="first query includes model loading")
        if ans.abstained:
            st.warning("Retrieval confidence is below the calibrated threshold: the sources may not answer this.")
        st.subheader("Answer, audited claim by claim")
        if not ans.verdicts:
            st.write(ans.raw_answer or "(no generator)")
        for i, v in enumerate(ans.verdicts, 1):
            color, desc = STATUS_STYLE[v.status]
            ref = f"[{v.support_ref}]" if v.support_ref else ("[new]" if v.new_chunk is not None else "")
            st.markdown(f"<div style='border-left:4px solid {color};padding:4px 10px;margin:4px 0'>"
                        f"{html.escape(v.text)} <b>{ref}</b><br><small style='color:{color}'>{v.status} - {desc}; "
                        f"cited {v.cited}; P(support)={v.p_support:.2f}, P(contradict)={v.p_contradict:.2f}</small></div>",
                        unsafe_allow_html=True)
            if v.support_sentence:
                with st.expander(f"evidence for claim {i}"):
                    st.write(v.support_sentence)
                    if v.new_chunk is not None:
                        ch = load_system(gen).index.chunks[v.new_chunk]
                        st.caption(f"new source: doc {ch.doc_id} - {ch.title}")
        with st.expander("raw generator output / edit it and re-audit (try adding 'not' or changing a number)"):
            edited = st.text_area("answer text with [n] citations", ans.raw_answer, height=120)
            if st.button("Re-audit edited answer"):
                # re-run the audit on the edited answer
                st.session_state["ans"] = load_system(gen).audit_text(r, edited)
                st.rerun()
        st.subheader("Sources")
        terms = set(r.terms)
        support_by_ref = {}
        for v in ans.verdicts:
            if v.support_ref:
                support_by_ref.setdefault(v.support_ref, v.support_sentence)
        for i, ev in enumerate(r.contexts, 1):
            row = r.ranked[i - 1]
            with st.expander(f"[{i}] {ev.title}  (doc {row['doc_id']}, LTR {row['ltr']:.2f})" if row["ltr"] is not None
                             else f"[{i}] {ev.title}"):
                st.markdown(highlight(ev.text, terms, support_by_ref.get(i, "")), unsafe_allow_html=True)

with tab_ir:
    ans = st.session_state.get("ans")
    if not ans:
        st.info("Run a query first.")
    else:
        r = ans.retrieval
        vt = load_system(gen)
        if r.out_of_scope:
            st.warning(f"Out-of-scope gate: best cross-encoder relevance {r.best_ce:.2f} < threshold {r.scope_tau:.2f} "
                       "(1st percentile of in-domain training questions) -> 0 documents returned. "
                       "The lists below are the closest matches that were rejected.")
        st.markdown("**Query analysis** (case folding, tokenisation, stop words, Porter stemming)")
        from veritrace.pipeline import explain_query_processing
        st.write(" ".join(f"`{a}→{b}`" for a, b in explain_query_processing(r.query)))
        st.dataframe(pd.DataFrame(r.term_stats, columns=["term", "df (chunks)", "idf", "title df"]), hide_index=True)
        st.markdown("**Postings of the query terms**")
        for t, df, idf, _ in r.term_stats:
            e = vt.index.describe_term(t)
            if e["df"]:
                with st.expander(f"{t}: df={e['df']}, idf={e['idf']}"):
                    st.json(e)
        full = vt.sparse.chunk_scores(r.query)
        if full.max() > 0:
            import numpy as np
            top_chunk = int(np.argmax(full))
            st.markdown(f"**Where the BM25 score of the top chunk comes from** (chunk {top_chunk}, doc "
                        f"{vt.index.chunks[top_chunk].doc_id}; score = (1-g)·body + g·title = {full[top_chunk]:.3f})")
            st.dataframe(pd.DataFrame(vt.sparse.explain(r.query, top_chunk)).round(3), hide_index=True)
            hits_t, tier, n_t = vt.sparse.search_tiered(r.query, r=100, k=10)
            from veritrace.sparse import heap_topk
            same = len({c for c, _ in heap_topk(full, 10)} & {c for c, _ in hits_t})
            st.caption(f"Tiered index (tier 1 = champion lists, r = 100): answered from tier {tier}, scored {n_t} "
                       f"chunks instead of {int((full > 0).sum())}; {same}/10 of the exhaustive top-10 found.")
        c1, c2 = st.columns(2)
        c1.markdown(f"**BM25 with learned zones** (k1={vt.sparse.k1}, b={vt.sparse.b}, g={vt.sparse.g})")
        c1.dataframe(pd.DataFrame(r.sparse_top, columns=["doc", "score"]), hide_index=True)
        c2.markdown("**Dense (BGE-small) cosine**")
        c2.dataframe(pd.DataFrame(r.dense_top, columns=["doc", "cosine"]), hide_index=True)
        st.markdown(f"**Router**: P(lexical wins) = {r.p_lexical:.3f} → a(q) = {r.alpha:.2f} "
                    f"(best fixed = {vt.alpha_fixed})")
        st.dataframe(pd.DataFrame([{**r.qpp, **r.ce_arb}]).T.rename(columns={0: "value"}))
        st.markdown("**LambdaMART re-ranking**")
        st.dataframe(pd.DataFrame(r.ranked[:15]), hide_index=True)

with tab_index:
    vt = load_system(gen)
    term = st.text_input("Look up a term in the dictionary", "methadone")
    if term:
        st.json(vt.index.describe_term(stem(term.lower())))
    bq = st.text_input("Boolean query", '"liver transplantation" AND methadone')
    if bq:
        res = vt.boolean.query(bq)
        st.write(f"parse tree: `{vt.boolean.parse(bq)}` → {len(res)} chunks")
        st.dataframe(pd.DataFrame([{"chunk": c, "doc": vt.index.chunks[c].doc_id, "title": vt.index.chunks[c].title}
                                   for c in res[:50]]), hide_index=True)

with tab_eval:
    for name in sorted(Path(FIGURES).glob("*.png")):
        st.image(str(name), caption=name.stem)
    p = Path(RESULTS) / "scifact_retrieval.json"
    if p.exists():
        st.dataframe(pd.DataFrame(json.load(open(p))["metrics"]).T)
