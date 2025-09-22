import time
import io
import base64
import os
import re
import asyncio
import pandas as pd
from tqdm.auto import tqdm

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# ======================
# 0) Settings
# ======================
POLICIES_PATH = "gov24_services_with_tags.csv"
CSV_PATH_2 = "gov24_services_with_tags (1).csv"
COLLECTION_NAME = "gov_services"
MODEL_NAME = "BAAI/bge-m3"
ENGINE_DEFAULT = "fw"
LANGUAGE = "ko"
FW_BEAM = 5
MAX_READ = 3
SUMMARY_CLIP = 300
DEFAULT_VOICE = "ko-KR-SunHiNeural"
CORS_ORIGINS = ["*"]

# ======================
# 1) Services
# ======================
class PolicySearch:
    def __init__(self, csv_path: str):
        self.df = None
        self.documents = None
        self.payloads = None
        self.model = None
        self.client = None
        
        if os.path.exists(csv_path):
            self.df, self.payloads, self.documents = self.load_and_prepare(csv_path)
            self.model, self.client = self.build_index(self.documents, self.payloads)
        elif os.path.exists(CSV_PATH_2):
            self.df, self.payloads, self.documents = self.load_and_prepare(CSV_PATH_2)
            self.model, self.client = self.build_index(self.documents, self.payloads)
        else:
            print(f"Error: Neither {csv_path} nor {CSV_PATH_2} were found.")

    def rows(self):
        return len(self.df) if self.df is not None else 0
        
    def load_and_prepare(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["서비스명"] = df["서비스명"].fillna("")
        df["tags"] = df["tags"].fillna("")
        df["지원내용"] = df["지원내용"].fillna("")
        df["combined_text"] = (df["서비스명"] + " " + df["tags"]) * 3 + " " + df["지원내용"]
        payloads = df.to_dict(orient="records")
        documents = df["combined_text"].tolist()
        return df, payloads, documents

    def build_index(self, documents, payloads):
        model = SentenceTransformer(MODEL_NAME)
        dim = model.get_sentence_embedding_dimension()
        client = QdrantClient(":memory:")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        vectors = model.encode(documents, show_progress_bar=False)
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(id=i, vector=v.tolist(), payload=p)
                for i, (v, p) in enumerate(zip(vectors, payloads))
            ],
            batch_size=256,
            wait=True,
        )
        return model, client

    def search(self, query_text: str, top_k: int = 3):
        if not self.model or not self.client:
            return []
        
        resp = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.model.encode(query_text).tolist(),
            limit=10,
        )
        hits = resp.points

        reranked = []
        query_keywords = query_text.split()

        for h in hits:
            payload = h.payload
            bonus = 0.0
            
            tags_text = str(payload.get("tags", ""))
            tag_list = [t.strip() for t in re.split(r"[,;\s]\s*", tags_text) if t]
            for kw in query_keywords:
                if kw in tag_list:
                    bonus += 0.5
            
            service_name = str(payload.get("서비스명", ""))
            for kw in query_keywords:
                if kw and kw in service_name:
                    bonus += 0.2

            support = str(payload.get("지원내용", ""))
            for kw in query_keywords:
                if kw and kw in support:
                    bonus += 0.05
            
            reranked.append({"hit": h, "final_score": (h.score or 0.0) + bonus})

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        
        return [item["hit"].payload for item in reranked[:top_k]]

class TTS:
    async def synth_mp3_bytes(self, text: str, voice: str = DEFAULT_VOICE) -> bytes:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        out_path = "_tmp_tts.mp3"
        await communicate.save(out_path)
        with open(out_path, "rb") as f:
            data = f.read()
        try:
            os.remove(out_path)
        except Exception:
            pass
        return data

    def build_spoken_text(self, items, pause_ms=400):
        parts = []
        for i, r in enumerate(items[:MAX_READ], 1):
            title = (r.get("서비스명") or "").strip()
            summary = (r.get("지원내용") or "").strip()
            if len(summary) > SUMMARY_CLIP:
                summary = summary[:SUMMARY_CLIP].rstrip() + " ..."
            parts.append(f"{i}번 정책은 {title} 입니다. 요약: {summary}")
        if not parts:
            return "적합한 정책을 찾지 못했습니다. 더 구체적으로 말씀해 주세요."
        return " ".join(parts)

# ======================
# 2) FastAPI 앱 초기화
# ======================
app = FastAPI(title="정책 검색 서버", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ASR 서비스는 별도로 가정하고, 여기서는 클래스만 정의합니다.
class FasterWhisperASR:
    def __init__(self):
        pass
    def transcribe_bytes(self, data, language):
        return "청년 주거 지원 정책을 찾고 있습니다", len(data) / 16000

class OpenAIWhisperASR:
    def __init__(self):
        pass
    def transcribe_bytes(self, data, language):
        return "청년 주거 지원 정책을 찾고 있습니다", len(data) / 16000

# ======================
# 3) 서버 시작 시 초기화
# ======================
@app.on_event("startup")
def startup_event():
    global SEARCH, TTS_SERVICE, FW, OW
    print("Initializing services...")
    SEARCH = PolicySearch(POLICIES_PATH)
    TTS_SERVICE = TTS()
    FW = FasterWhisperASR()
    OW = OpenAIWhisperASR()
    print("Services initialized.")

# ======================
# 4) 엔드포인트
# ======================
@app.get("/healthz")
def healthz():
    ok = True
    notes = []
    if SEARCH.rows() == 0:
        ok = False
        notes.append("정책 CSV 비었거나 경로 불일치")
    return {"ok": ok, "policy_rows": SEARCH.rows(), "policies_path": POLICIES_PATH, "notes": notes}

@app.post("/stt_search_tts")
async def stt_search_tts(
    audio: UploadFile = File(...),
    engine: str = Form(ENGINE_DEFAULT),
    language: str = Form(LANGUAGE),
    beam_size: int = Form(FW_BEAM),
    topk: int = Form(3),
    voice: str = Form(DEFAULT_VOICE),
):
    raw = await audio.read()
    t0 = time.time()

    if engine == "fw":
        text, dur = FW.transcribe_bytes(raw, language=language)
    else:
        text, dur = OW.transcribe_bytes(raw, language=language)
    stt_time = time.time() - t0

    results = SEARCH.search(text, topk=int(topk))
    if results:
        best = results[0]
        spoken = f"추천 정책은 {best.get('서비스명','') or ''} 입니다. 요약: {best.get('지원내용','') or ''}"
    else:
        spoken = "적합한 정책을 찾지 못했습니다. 더 구체적으로 말씀해 주세요."

    t1 = time.time()
    mp3_bytes = await TTS_SERVICE.synth_mp3_bytes(spoken, voice=voice)
    tts_time = time.time() - t1

    audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")

    return JSONResponse({
        "stt": {"text": text, "engine": engine, "audio_sec": dur, "decode_s": stt_time, "language": language},
        "search": {"query": text, "topk": int(topk), "results": results},
        "tts": {"voice": voice, "spoken_text": spoken, "synthesis_s": tts_time, "audio_mp3_b64": audio_b64},
        "pipeline_ms": int((time.time() - t0) * 1000)
    })
