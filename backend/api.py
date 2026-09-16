"""
MedBuddy RAG API Server

Bridges the static frontend (GitHub Pages) with FAISS retrieval + LLM API.
Must be run in the same directory as query.py, with faiss_index/ and bge_model/ present.

Run locally:
    pip install fastapi uvicorn openai sentence-transformers faiss-cpu numpy
    ARK_API_KEY=ark-xxx uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health       — liveness check
    POST /api/chat     — RAG Q&A

Supported providers (checked in order):
    ARK_API_KEY        — 火山方舟 (Volcengine), model: doubao-1-5-lite-32k (cheap/free)
    SILICONFLOW_API_KEY — SiliconFlow, model: Qwen/Qwen2.5-7B-Instruct
    DEEPSEEK_API_KEY   — DeepSeek official API
    GEMINI_API_KEY     — Google Gemini 2.0 Flash
    ANTHROPIC_API_KEY  — Claude (paid)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from query import load_index_and_meta, retrieve, build_context
from onnx_query import load_bge_model

app = FastAPI(title="MedBuddy RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hazel2508.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:4173",
        "null",
    ],
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


# ── Medication name normalization ─────────────────────────────────────────────

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

ENGLISH_GENERIC_NAMES = {
    "metformin", "glipizide", "dapagliflozin", "sitagliptin", "amlodipine",
    "lisinopril", "losartan", "hydrochlorothiazide", "atorvastatin",
    "rosuvastatin", "metoprolol", "aspirin", "levothyroxine", "albuterol",
    "alendronate",
}


def find_english_drugs(drug_names: list[str]) -> list[str]:
    found = set()
    for name in drug_names:
        normalized = name.lower()
        for generic in ENGLISH_GENERIC_NAMES:
            if generic in normalized:
                found.add(generic)
        for zh, en in DRUG_MAP.items():
            if zh in name:
                found.add(en)
    return list(found)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the medication information assistant in MedBuddy, a chronic-care medication management app for patients in the United States.
Answer in clear American English using only the FDA drug label excerpts provided in the conversation.

Rules:
1. Base every claim only on the provided label excerpts. Do not add information that is not supported by the excerpts.
2. Cite the relevant medication and label section, for example: "According to the [Section] section of the [Medication] label..."
3. If the excerpts do not contain enough information, say: "This question is outside the information available in the provided FDA label excerpts. Please ask your clinician or pharmacist."
4. End every answer with this reminder: "Before changing how you take any medication, talk with your prescriber or a licensed pharmacist."
5. Use a warm, concise, patient-friendly tone. Explain necessary medical terms in plain language.
"""


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    drug_names: list[str] = []   # Medication names currently saved in the app
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

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "..", "frontend", "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "vectors": _index.ntotal if _index is not None else 0,
    }


# ── LLM call (provider-agnostic) ───────────────────────────────────────────────

def call_llm(messages: list[dict], system: str, max_tokens: int) -> str:
    """messages: [{role: 'user'|'assistant', content: str}, ...]"""
    if os.environ.get("ARK_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["ARK_API_KEY"],
                        base_url="https://ark.cn-beijing.volces.com/api/v3")
        resp = client.chat.completions.create(
            model="doubao-1-5-lite-32k-250115",
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content
    if os.environ.get("SILICONFLOW_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["SILICONFLOW_API_KEY"],
                        base_url="https://api.siliconflow.cn/v1")
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content
    if os.environ.get("DEEPSEEK_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                        base_url="https://api.deepseek.com/v1")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content
    if os.environ.get("GEMINI_API_KEY"):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        history = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])])
                   for m in messages]
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
        )
        return resp.text
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-opus-4-8", max_tokens=max_tokens, system=system, messages=messages,
        )
        return response.content[0].text
    raise HTTPException(500, detail="Set ARK_API_KEY / SILICONFLOW_API_KEY / DEEPSEEK_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY")


def translate_to_english(question: str) -> str:
    """Keep English queries unchanged; translate non-ASCII queries for BGE retrieval."""
    if question.isascii():
        return question
    try:
        return call_llm(
            [{"role": "user", "content": f"Translate to English, output only the translation:\n{question}"}],
            system="You are a translator.", max_tokens=200,
        ).strip()
    except Exception:
        return question


def has_llm_provider() -> bool:
    return any(os.environ.get(key) for key in (
        "ARK_API_KEY",
        "SILICONFLOW_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
    ))


def build_retrieval_fallback(chunks: list[dict]) -> str:
    """Return grounded FDA label excerpts when no generation provider is configured."""
    if not chunks:
        return (
            "I could not find a relevant passage in the available FDA drug labels. "
            "Please ask your prescriber or pharmacist.\n\n"
            "Before changing how you take any medication, talk with your prescriber "
            "or a licensed pharmacist."
        )

    excerpts = []
    for chunk in chunks[:3]:
        text = " ".join(chunk.get("text", "").split())
        if len(text) > 500:
            text = text[:497].rstrip() + "..."
        excerpts.append(f"{chunk['drug'].title()}, {chunk['section']}:\n{text}")

    return (
        "I found these relevant passages in the FDA drug labels:\n\n"
        + "\n\n".join(excerpts)
        + "\n\nBefore changing how you take any medication, talk with your "
          "prescriber or a licensed pharmacist."
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, detail="question is empty")

    # Normalize medication names; use a single-drug filter when unambiguous
    en_drugs = find_english_drugs(req.drug_names)
    drug_filter = en_drugs[0] if len(en_drugs) == 1 else None

    # BGE is English-only; non-English questions are translated before retrieval
    query_en = translate_to_english(req.question)

    # RAG retrieval
    chunks = retrieve(query_en, _model, _index, _metadata,
                      top_k=req.top_k, drug_filter=drug_filter)

    context = build_context(chunks) if chunks else ""
    sources = [SourceItem(drug=c["drug"], section=c["section"], score=round(c["_score"], 3))
               for c in chunks]

    # Assemble user message (RAG context + question + patient drug list)
    user_msg = ""
    if context:
        user_msg = (
            "Relevant excerpts retrieved from official FDA drug labels:\n"
            "──────────────────────────────────────────────\n"
            f"{context}\n"
            "──────────────────────────────────────────────\n\n"
        )
    user_msg += f"Patient question: {req.question}"
    if req.drug_names:
        user_msg += f"\n\nMedications currently saved in MedBuddy: {' / '.join(req.drug_names)}"

    # Build message list with conversation history
    messages = [m for m in req.history if m.get("role") in ("user", "assistant")][-8:]
    messages.append({"role": "user", "content": user_msg})

    if has_llm_provider():
        answer_text = call_llm(messages, system=SYSTEM_PROMPT, max_tokens=1024)
    else:
        answer_text = build_retrieval_fallback(chunks)

    return ChatResponse(answer=answer_text, sources=sources)
