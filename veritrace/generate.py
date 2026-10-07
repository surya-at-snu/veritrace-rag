from __future__ import annotations

import os
import re
import time

import numpy as np

from .config import LLM_MODEL, LLM_MODEL_SMALL

# the llm may only use the numbered sources and has to cite one after every sentence
SYSTEM_PROMPT = (
    "You are VeriTrace, a careful scientific assistant. Answer using ONLY the numbered sources. "
    "Write 3 or 4 sentences. Each sentence must state one specific finding from the sources (what was "
    "studied, in whom, the direction or size of the effect) and must end with the number of the source "
    "that supports it in square brackets, for example [2]. Do not use outside knowledge. If the sources "
    "do not contain the answer, reply exactly: INSUFFICIENT EVIDENCE."
)


def is_claim(text: str) -> bool:
    t = text.strip()
    return not t.endswith("?") and not re.match(r"^(what|why|how|when|where|which|who|whom|whose|is|are|does|do|can|could|should|will|would|did|was|were)\b", t, re.I)


def format_sources(contexts, max_words: int = 160) -> str:
    lines = []
    for i, ev in enumerate(contexts, 1):
        words = ev.text.split()
        body = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        lines.append(f"[{i}] {ev.title}\n{body}")
    return "\n\n".join(lines)


def build_user_prompt(query: str, contexts) -> str:
    src = format_sources(contexts)
    # claims get a support / contradict style prompt, questions are asked as they are
    if is_claim(query):
        task = (f"Claim: {query}\n\nDo the sources support or contradict this claim? Answer in 3 or 4 "
                f"sentences that state what the sources report, citing a source after every sentence.")
    else:
        task = f"Question: {query}"
    return f"Sources:\n\n{src}\n\n{task}"


class QwenGenerator:
    def __init__(self, model_name: str = LLM_MODEL, max_new_tokens: int = 220):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
        self.max_new_tokens = max_new_tokens
        self.name = model_name.split("/")[-1]

    def __call__(self, query: str, contexts) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, contexts)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            # greedy decoding so the answers are reproducible
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatGenerator:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or os.environ.get("VT_LLM_BASE_URL"),
                             api_key=api_key or os.environ.get("VT_LLM_API_KEY", "none"))
        self.model = model or os.environ.get("VT_LLM_MODEL", "gpt-4o-mini")
        self.name = self.model

    def __call__(self, query: str, contexts) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_user_prompt(query, contexts)}])
        return r.choices[0].message.content.strip()


class ExtractiveGenerator:

    def __init__(self, index, n_sentences: int = 3, lam: float = 0.7):
        from .dense import load_encoder
        from .verify import _VSM
        self.vsm = _VSM(index)
        self.enc = load_encoder()
        self.n, self.lam = n_sentences, lam
        self.name = "extractive-mmr"

    def __call__(self, query: str, contexts) -> str:
        from .text import analyze
        cands = [(i, s) for i, ev in enumerate(contexts, 1) for s in ev.sentences if len(s.split()) >= 6]
        if not cands:
            return "INSUFFICIENT EVIDENCE"
        q = self.vsm.qvec(analyze(query))
        lex = np.array([self.vsm.cos(q, self.vsm.dvec(analyze(s))) for _, s in cands])
        E = self.enc.encode([s for _, s in cands] + [query], normalize_embeddings=True, show_progress_bar=False)
        den = E[:-1] @ E[-1]
        rel = lex / (lex.max() + 1e-9) + den / (den.max() + 1e-9)
        chosen = []
        while len(chosen) < min(self.n, len(cands)):
            red = np.array([max((float(E[i] @ E[j]) for j in chosen), default=0.0) for i in range(len(cands))])
            # mmr: relevant to the query but not a repeat of what we already picked
            mmr = self.lam * rel - (1 - self.lam) * red * 2
            mmr[chosen] = -1e9
            chosen.append(int(np.argmax(mmr)))
        chosen.sort(key=lambda i: (cands[i][0], i))
        return " ".join(f"{cands[i][1].rstrip()} [{cands[i][0]}]" for i in chosen)


def make_generator(kind: str, index=None):
    if kind in ("qwen", "qwen1.5b"):
        return QwenGenerator(LLM_MODEL)
    if kind == "qwen0.5b":
        return QwenGenerator(LLM_MODEL_SMALL)
    if kind == "openai":
        return OpenAICompatGenerator()
    if kind == "extractive":
        return ExtractiveGenerator(index)
    raise ValueError(kind)


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, time.time() - t
