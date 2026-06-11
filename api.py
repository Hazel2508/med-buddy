"""
MedBuddy RAG API Server

Bridges the static frontend (GitHub Pages) with FAISS retrieval + Claude API.
Must be run in the same directory as query.py, with faiss_index/ and bge_model/ present.

Run locally:
    pip install fastapi uvicorn anthropic
    ANTHROPIC_API_KEY=sk-ant-... uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health       — liveness check
    POST /api/chat     — RAG Q&A
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from query import load_index_and_meta, load_bge_model, retrieve, build_context

app = FastAPI(title="MedBuddy RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Startup: load heavy resources once ───────────────────────────────────────

_index = _metadata = _model = None


@app.on_event("startup")
def startup():
    global _index, _metadata, _model
    print("Loading FAISS index...", flush=True)
    _index, _metadata = load_index_and_meta()
    print(f"  {_index.ntotal} vectors", flush=True)
    print("Loading BGE model...", flush=True)
    _model = load_bge_model()
    print("  Ready.", flush=True)


# ── Chinese drug name → English generic mapping ───────────────────────────────

DRUG_MAP = {
    "二甲双胍":  "metformin",
    "格列吡嗪":  "glipizide",
    "达格列净":  "dapagliflozin",
    "西格列汀":  "sitagliptin",
    "氨氯地平":  "amlodipine",
    "赖诺普利":  "lisinopril",
    "氯沙坦":    "losartan",
    "氢氯噻嗪":  "hydrochlorothiazide",
    "阿托伐他汀": "atorvastatin",
    "瑞舒伐他汀": "rosuvastatin",
    "美托洛尔":  "metoprolol",
    "阿司匹林":  "aspirin",
    "左甲状腺素": "levothyroxine",
    "沙丁胺醇":  "albuterol",
    "阿仑膦酸":  "alendronate",
}


def find_english_drugs(chinese_names: list[str]) -> list[str]:
    found = set()
    for name in chinese_names:
        for zh, en in DRUG_MAP.items():
            if zh in name:
                found.add(en)
    return list(found)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是MedBuddy慢病用药管理应用的专业AI助手。
你的回答必须基于下方提供的FDA官方药品说明书原文（英文），用清晰的简体中文回答患者的用药问题。

回答规则：
1. 仅基于提供的说明书原文内容作答，不要补充说明书以外的信息
2. 引用原文时注明来源，例如"根据[药名]说明书的[章节]部分"
3. 说明书中没有相关内容时，明确告知"该问题超出说明书范围，建议咨询医生或药师"
4. 每条回答结尾必须提醒患者：用药方案的任何调整，请先咨询您的主治医生或执业药师
5. 语气温和、清晰，适合中老年慢性病患者，避免过多医学术语
"""


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    drug_names: list[str] = []   # Chinese drug names from the app (e.g. "阿托伐他汀钙片")
    history: list[dict] = []     # recent turns: [{role, content}, ...]
    top_k: int = 5


class SourceItem(BaseModel):
    drug: str
    section: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "vectors": _index.ntotal if _index is not None else 0,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, detail="ANTHROPIC_API_KEY not set on server")
    if not req.question.strip():
        raise HTTPException(400, detail="question is empty")

    # Map Chinese drug names → English; use single-drug filter when unambiguous
    en_drugs = find_english_drugs(req.drug_names)
    drug_filter = en_drugs[0] if len(en_drugs) == 1 else None

    # RAG retrieval
    chunks = retrieve(req.question, _model, _index, _metadata,
                      top_k=req.top_k, drug_filter=drug_filter)

    context = build_context(chunks) if chunks else ""
    sources = [SourceItem(drug=c["drug"], section=c["section"], score=round(c["_score"], 3))
               for c in chunks]

    # Assemble user message (RAG context + question + patient drug list)
    user_msg = ""
    if context:
        user_msg = (
            "以下是从FDA官方药品说明书中检索到的相关原文：\n"
            "──────────────────────────────────────────────\n"
            f"{context}\n"
            "──────────────────────────────────────────────\n\n"
        )
    user_msg += f"患者问题：{req.question}"
    if req.drug_names:
        user_msg += f"\n\n患者当前用药：{'  /  '.join(req.drug_names)}"

    # Build message list with conversation history
    messages = [m for m in req.history if m.get("role") in ("user", "assistant")][-8:]
    messages.append({"role": "user", "content": user_msg})

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return ChatResponse(answer=response.content[0].text, sources=sources)
