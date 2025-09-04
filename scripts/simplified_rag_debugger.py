"""
simplified_rag_debugger.py
==============================

Lightweight harness for inspecting your RAG pipeline end-to-end, with
optional OpenAI generation controlled via .env and CLI arguments.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Make local modules importable
import sys
sys.path.append(str(Path(__file__).resolve().parent))

# Try your project import first; fall back to local file.
try:
    from rag.generation.prompt_constructor import DynamicPromptConstructor  # type: ignore
except Exception:
    from prompt_constructor import DynamicPromptConstructor  # type: ignore

# Load environment variables (.env in project root or current dir)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    contents: str
    meta: Dict[str, Any] = field(default_factory=dict)

    sparse_score: float = 0.0
    field_score: float = 0.0
    semantic_score: float = 0.0
    fusion_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = {"id": self.id, "contents": self.contents, "meta": self.meta}
        if self.sparse_score: data["sparse_score"] = self.sparse_score
        if self.field_score: data["field_score"] = self.field_score
        if self.semantic_score: data["semantic_score"] = self.semantic_score
        if self.fusion_score: data["fusion_score"] = self.fusion_score
        return data


# ---------------------------------------------------------------------------
# Query parsing (simple, deterministic)
# ---------------------------------------------------------------------------

def simple_parse_query(query: str) -> Dict[str, Any]:
    q = query.lower()
    result: Dict[str, Any] = {}

    # Duration
    m = re.search(r"(\d{1,3})\s*[- ]?\s*(min|mins|minute|minutes)", q)
    if m:
        try:
            result["duration"] = int(m.group(1))
        except ValueError:
            pass
    else:
        m = re.search(r"(\d{1,2})\s*h\s*(\d{1,2})", q)
        if m:
            result["duration"] = int(m.group(1)) * 60 + int(m.group(2))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)", q)
            if m:
                result["duration"] = int(round(float(m.group(1)) * 60))

    # Participants
    if any(term in q for term in ["solo", "by myself", "alone on court"]):
        result["participants"] = 1
    else:
        m = re.search(r"(\d{1,2})\s*(player|players|participants)", q)
        if m:
            try:
                result["participants"] = int(m.group(1))
            except ValueError:
                pass

    # Type
    if "conditioned game" in q:
        result["type"] = "conditioned game"
    elif "ghosting" in q:
        result["type"] = "ghosting"
    elif "drill" in q:
        result["type"] = "drill"
    elif "solo" in q:
        result["type"] = "solo practice"
    elif "game" in q:
        result["type"] = "game"

    return result


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> List[Document]:
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    docs: List[Document] = []

    # Try JSON list
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, entry in enumerate(data):
                if not isinstance(entry, dict): continue
                doc_id = str(entry.get("id") or entry.get("session_id") or i)
                contents = entry.get("contents") or entry.get("context") or ""
                meta = entry.get("meta") or {}
                docs.append(Document(id=doc_id, contents=contents, meta=meta))
            return docs
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = str(entry.get("id") or entry.get("session_id") or i)
            contents = entry.get("contents") or entry.get("context") or ""
            meta = entry.get("meta") or {}
            docs.append(Document(id=doc_id, contents=contents, meta=meta))
    return docs


# ---------------------------------------------------------------------------
# Sparse retrieval (TF–IDF)
# ---------------------------------------------------------------------------

def sparse_retrieval(query: str, corpus: List[Document], top_k: int = 10) -> List[Document]:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([d.contents for d in corpus])
    qv = vectorizer.transform([query])
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    scored = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    out: List[Document] = []
    for doc, score in scored:
        nd = Document(id=doc.id, contents=doc.contents, meta=doc.meta.copy())
        nd.sparse_score = float(score)
        out.append(nd)
    return out


# ---------------------------------------------------------------------------
# Field retrieval (very simple, metadata-based)
# ---------------------------------------------------------------------------

def field_retrieval(query_fields: Dict[str, Any], corpus: List[Document], top_k: int = 10) -> List[Document]:
    scored: List[tuple[Document, float]] = []
    for doc in corpus:
        s = 0.0
        meta = doc.meta or {}

        # duration
        qd = query_fields.get("duration")
        if qd is not None:
            dv = meta.get("duration") or meta.get("Duration")
            try:
                if dv is not None and int(dv) == int(qd):
                    s += 1.0
            except (ValueError, TypeError):
                pass

        # participants
        qp = query_fields.get("participants")
        if qp is not None:
            dp = meta.get("participants") or meta.get("num_players")
            try:
                if dp is not None and int(dp) == int(qp):
                    s += 1.0
            except (ValueError, TypeError):
                pass

        # type
        qt = query_fields.get("type")
        if qt:
            dt = meta.get("type") or meta.get("session_type")
            if isinstance(dt, str) and dt.lower() == qt.lower():
                s += 1.0
            if qt.lower() in doc.contents.lower():
                s += 0.5

        if s > 0:
            scored.append((doc, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    out: List[Document] = []
    for doc, s in scored[:top_k]:
        nd = Document(id=doc.id, contents=doc.contents, meta=doc.meta.copy())
        nd.field_score = float(s)
        out.append(nd)
    return out


# ---------------------------------------------------------------------------
# Fusion (weighted RRF)
# ---------------------------------------------------------------------------

def rrf_fusion(ranked_lists: Dict[str, List[Document]], query_fields: Dict[str, Any],
               top_k: int = 10, k_constant: int = 60) -> List[Document]:
    is_specific = len(query_fields) > 0
    if is_specific:
        weights = {"sparse": 0.45, "field": 0.45, "semantic": 0.10}
    else:
        weights = {"sparse": 0.20, "field": 0.20, "semantic": 0.60}

    scores: Dict[str, float] = defaultdict(float)
    registry: Dict[str, Document] = {}

    for name, docs in ranked_lists.items():
        w = weights.get(name, 0.0)
        if w <= 0 or not docs:
            continue
        for rank, doc in enumerate(docs):
            scores[doc.id] += w * (1.0 / (k_constant + rank + 1))
            if doc.id not in registry:
                registry[doc.id] = doc
            else:
                ex = registry[doc.id]
                ex.sparse_score = max(ex.sparse_score, doc.sparse_score)
                ex.field_score = max(ex.field_score, doc.field_score)
                ex.semantic_score = max(ex.semantic_score, doc.semantic_score)

    fused: List[Document] = []
    for doc_id, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        d = registry[doc_id]
        d.fusion_score = s
        fused.append(d)
    return fused


# ---------------------------------------------------------------------------
# Prompt + (optional) LLM answer
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str) -> str:
    """
    Call OpenAI using either the new (>=1.x) SDK or legacy SDK.
    Returns model output or raises if both fail.
    """
    # New SDK
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Legacy SDK
        try:
            import openai  # type: ignore
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed (both clients). Details: {e}")


def build_prompt_and_answer(query: str, retrieved_docs: List[Document],
                            *, model: Optional[str] = None, use_llm: bool = False) -> Tuple[str, str]:
    docs_for_prompt = [{"id": d.id, "contents": d.contents, "meta": d.meta} for d in retrieved_docs]
    dpc = DynamicPromptConstructor()
    prompt = dpc.create_prompt(query=query, retrieved_docs=docs_for_prompt)

    selected_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if use_llm:
        if not os.getenv("OPENAI_API_KEY"):
            answer = "[OPENAI_API_KEY not found in environment; returning stub answer.]"
        else:
            try:
                answer = call_llm(prompt, selected_model)
            except Exception as e:
                answer = f"[LLM call failed: {e}]"
    else:
        answer = "[LLM call disabled; returning stub answer.]"

    return prompt, answer


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_demo(query: str, corpus_path: str, top_k: int = 5,
             *, model: Optional[str] = None, use_llm: bool = False) -> None:
    print("\n=== Query ===")
    print(query)

    # 1) Parse
    qf = simple_parse_query(query)
    print("\n--- Parsed query fields ---")
    for k, v in qf.items(): print(f"{k}: {v}")
    if not qf: print("(No fields extracted; query will be treated as vague.)")

    # 2) Load corpus
    print("\n--- Loading corpus ---")
    corpus = load_corpus(corpus_path)
    print(f"Loaded {len(corpus)} documents from {corpus_path}")

    # 3) Sparse
    print("\n--- Sparse retrieval (TF–IDF) ---")
    sparse_docs = sparse_retrieval(query, corpus, top_k=top_k)
    for i, d in enumerate(sparse_docs, 1):
        print(f"{i:2d}. id={d.id}, sparse_score={d.sparse_score:.4f}")

    # 4) Field
    print("\n--- Field retrieval (rudimentary) ---")
    field_docs = field_retrieval(qf, corpus, top_k=top_k)
    if field_docs:
        for i, d in enumerate(field_docs, 1):
            print(f"{i:2d}. id={d.id}, field_score={d.field_score:.4f}")
    else:
        print("No documents matched the query fields.")

    # 5) Fusion
    print("\n--- RRF fusion ---")
    fused_docs = rrf_fusion({"sparse": sparse_docs, "field": field_docs, "semantic": []}, qf, top_k=top_k)
    if fused_docs:
        for i, d in enumerate(fused_docs, 1):
            bits = []
            if d.sparse_score: bits.append(f"sparse={d.sparse_score:.4f}")
            if d.field_score: bits.append(f"field={d.field_score:.4f}")
            if d.semantic_score: bits.append(f"semantic={d.semantic_score:.4f}")
            print(f"{i:2d}. id={d.id}, fusion_score={d.fusion_score:.4f}, " + ", ".join(bits))
    else:
        print("No documents were returned after fusion.")

    # 6) Prompt + Answer
    print("\n--- Dynamic prompt ---")
    prompt, answer = build_prompt_and_answer(query, fused_docs, model=model, use_llm=use_llm)
    print(prompt)
    print("\n--- Answer ---")
    print(f"[model={model or os.getenv('OPENAI_MODEL') or 'gpt-4o-mini'}] {answer}")


def main():
    """
    CLI entry point. You can also set defaults via environment variables:
      - OPENAI_API_KEY: your API key loaded via .env
      - OPENAI_MODEL  : default model if --model is not provided
    """
    import argparse

    parser = argparse.ArgumentParser(description="Simplified RAG debugger")
    parser.add_argument("--query", type=str, default="A 45-minute conditioned game", help="Query to run")
    parser.add_argument("--corpus", type=str, default="data/processed/balanced_grammar/balanced_500.jsonl", help="Path to JSON/JSONL corpus")
    parser.add_argument("--top-k", type=int, default=5, help="Top K per retriever and after fusion")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model to use if --use-llm is set")
    parser.add_argument("--use-llm", action="store_true", help="If set, call OpenAI with the constructed prompt")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"⚠️  The corpus file '{args.corpus}' does not exist. Please set --corpus to a valid JSON/JSONL corpus.")
        return

    run_demo(args.query, args.corpus, top_k=args.top_k, model=args.model, use_llm=args.use_llm)


if __name__ == "__main__":
    main()
