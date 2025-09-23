import time
import io
import base64
import os
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from rich.console import Console
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import edge_tts
import re
from konlpy.tag import Okt
# ======================
# 설정
# ======================
CSV_PATH = "gov24_services_with_tags (1).csv"
COLLECTION_NAME = "gov_services"
MODEL_NAME = "BAAI/bge-m3"
DEFAULT_VOICE = "ko-KR-SunHiNeural"
console = Console()
# -----------------------------
# 정책 검색 & TTS 서비스 클래스
# -----------------------------
class PolicySearch:
    def __init__(self, csv_path: str):
        self.df = None
        self.payloads = None
        self.documents = None
        self.model = None
        self.client = None
        self.is_ready = False
        try:
            self.load_and_prepare(csv_path)
            self.build_index()
            self.is_ready = True
        except Exception as e:
            console.print(f"[bold red]오류: PolicySearch 초기화 실패![/]")
            console.print_exception()
            self.is_ready = False
    def load_and_prepare(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 파일 경로를 찾을 수 없습니다: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df["서비스명"] = self.df["서비스명"].fillna("")
        self.df["tags"] = self.df["tags"].fillna("")
        self.df["지원내용"] = self.df["지원내용"].fillna("")
        self.df["combined_text"] = (self.df["서비스명"] + " " + self.df["tags"]) * 3 + " " + self.df["지원내용"]
        self.payloads = self.df.to_dict(orient="records")
        self.documents = self.df["combined_text"].tolist()
        console.print(f"[bold green]총 {len(self.documents)}개의 서비스 로드[/]")
    def build_index(self):
        console.print(f"[bold]임베딩 모델 로드:[/] {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        dim = self.model.get_sentence_embedding_dimension()
        self.client = QdrantClient(":memory:")
        self.client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        console.print("임베딩 생성 중...")
        vectors = self.model.encode(self.documents, show_progress_bar=False)
        console.print("Qdrant 업로드 중...")
        self.client.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(id=i, vector=v.tolist(), payload=p)
                for i, (v, p) in enumerate(zip(vectors, self.payloads))
            ],
            batch_size=256,
            wait=True,
        )
        console.print("[bold green]업로드 완료![/]")
    def search(self, query_text: str, topk: int = 3):
        try:
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
            return [item["hit"].payload for item in reranked[:topk]]
        except Exception as e:
            console.print(f"[bold red]오류: 검색 중 문제 발생: {e}[/]")
            return []
class EdgeTTS:
    async def synth(self, text: str, voice: str = DEFAULT_VOICE) -> bytes:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data") or chunk.get("content")
                if data:
                    audio_chunks.append(data)
        return b"".join(audio_chunks)
class FasterWhisperASR:
    def __init__(self, model_dir, compute_type, beam_size): pass
    def transcribe_bytes(self, data, language): return "청년 주거 지원 정책", 1.0
class OpenAIWhisperASR:
    def __init__(self, model_dir): pass
    def transcribe_bytes(self, data, language): return "청년 주거 지원 정책", 1.0
# ======================
# FastAPI 앱 초기화
# ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ======================
# 전역 인스턴스
# ======================
SEARCH = None
FW = None
OW = None
TTS_SYNTH = EdgeTTS()
@app.on_event("startup")
def startup_event():
    global SEARCH, FW, OW
    console.print("[bold yellow]서버 시작 중...[/]")
    SEARCH = PolicySearch(CSV_PATH)
@app.get("/healthz")
def healthz():
    ok = SEARCH and SEARCH.is_ready
    notes = []
    if not ok:
        notes.append("PolicySearch 인덱스가 준비되지 않았습니다.")
    return {"ok": ok, "notes": notes}
@app.post("/stt_search_tts")
async def stt_search_tts(
    audio: UploadFile = File(...),
    engine: str = Form("fw"),
    language: str = Form("ko"),
    beam_size: int = Form(5),
    topk: int = Form(3),
    voice: str = Form(DEFAULT_VOICE),
):
    try:
        raw = await audio.read()
        stt_start_time = time.time()
        text = "청년 주거 지원 정책을 찾고 있습니다"
        dur = len(raw) / 16000
        stt_time = time.time() - stt_start_time
        if not SEARCH or not SEARCH.is_ready:
            raise RuntimeError("정책 검색 서비스가 준비되지 않았습니다.")
        results = SEARCH.search(text, int(topk))
        if results:
            spoken_text = f"추천 정책은 {results[0].get('서비스명','') or 'N/A'} 입니다. 요약: {results[0].get('지원내용','') or 'N/A'}"
        else:
            spoken_text = "적합한 정책을 찾지 못했습니다. 더 구체적으로 말씀해 주세요."
        tts_start_time = time.time()
        mp3_bytes = await TTS_SYNTH.synth(spoken_text, voice=voice)
        tts_time = time.time() - tts_start_time
        def json_safe(x):
            import math
            if isinstance(x, float):
                if math.isnan(x) or math.isinf(x):
                    return None
                return x
            if isinstance(x, dict):
                return {k: json_safe(v) for k, v in x.items()}
            if isinstance(x, list):
                return [json_safe(v) for v in x]
            return x
        audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
        stt_obj = json_safe({"text": text, "engine": engine, "audio_sec": dur, "decode_s": stt_time, "language": language})
        search_obj = json_safe({"query": text, "topk": int(topk), "results": results})
        tts_obj = json_safe({"voice": voice, "spoken_text": spoken_text, "synthesis_s": tts_time, "audio_mp3_b64": audio_b64})
        return JSONResponse({
            "stt": stt_obj,
            "search": search_obj,
            "tts": tts_obj,
            "pipeline_ms": int((time.time() - stt_start_time) * 1000)
        })
    except Exception as e:
        console.print_exception()
        return JSONResponse(
            {"error": f"처리 중 오류 발생: {str(e)}"},
            status_code=500
        )
