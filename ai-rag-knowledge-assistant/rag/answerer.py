from __future__ import annotations

from typing import Dict, List

from openai import OpenAI

from .config import Settings
from .retriever import retrieve


def _fallback_answer(question: str, contexts: List[dict]) -> str:
    if not contexts:
        return "No context found. Please upload documents and rebuild the index."
    lines = [f"Question: {question}", "", "Most relevant context:"]
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] ({c['source']}) {c['text']}")
    return "\n".join(lines)


def _llm_answer(question: str, contexts: List[dict], settings: Settings) -> str:
    if not settings.openai_api_key:
        return _fallback_answer(question, contexts)

    ctx = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in contexts])
    system = (
        "You are a helpful assistant. Answer the question ONLY using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def answer_question(question: str, index: Dict, top_k: int, settings: Settings, use_llm: bool) -> Dict:
    contexts = retrieve(question, index, top_k=top_k, settings=settings)
    if use_llm:
        ans = _llm_answer(question, contexts, settings)
        used_llm = bool(settings.openai_api_key)
    else:
        ans = _fallback_answer(question, contexts)
        used_llm = False
    return {"answer": ans, "contexts": contexts, "used_llm": used_llm}
