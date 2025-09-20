# -*- coding: utf-8 -*-
"""
정책 CSV → Qdrant(in-memory) 인덱스 → 텍스트 쿼리로 검색 → Top3를 Edge-TTS로 읽어 MP3 저장
필요 패키지:
  pip install sentence-transformers qdrant-client rich edge-tts pandas tqdm
"""

import os
import re
import asyncio
import base64
import pandas as pd
from tqdm.auto import tqdm
from rich.console import Console
from rich.table import Table

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import edge_tts

# -----------------------------
# 0) 설정
# -----------------------------
CSV_PATH = "gov24_services_with_tags (1).csv"  # CSV 경로
COLLECTION_NAME = "gov_services"
MODEL_NAME = "BAAI/bge-m3"
MAX_READ = 3  # TTS로 읽을 최대 정책 개수
SUMMARY_CLIP = 300  # 지원내용 길이 제한(너무 길면 잘라서 읽음)
DEFAULT_VOICE = "ko-KR-SunHiNeural"

console = Console()

# -----------------------------
# 1) 데이터 로드 & 전처리
# -----------------------------
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    # 결측치 안전 처리
    df["서비스명"] = df["서비스명"].fillna("")
    df["tags"] = df["tags"].fillna("")
    df["지원내용"] = df["지원내용"].fillna("")
    # 벡터화 텍스트: (서비스명+tags)*3 + 지원내용
    df["combined_text"] = (df["서비스명"] + " " + df["tags"]) * 3 + " " + df["지원내용"]
    payloads = df.to_dict(orient="records")
    documents = df["combined_text"].tolist()
    console.print(f"[bold green]총 {len(documents)}개의 서비스 로드[/] (예시 텍스트: {documents[0][:80]}...)")
    return df, payloads, documents

# -----------------------------
# 2) 임베딩 모델 & Qdrant(in-memory)
# -----------------------------
def build_index(documents, payloads):
    console.print(f"[bold]임베딩 모델 로드:[/] {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    console.print(f"임베딩 차원: {dim}")

    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    console.print("임베딩 생성 중...")
    vectors = model.encode(documents, show_progress_bar=True)

    console.print("Qdrant 업로드 중...")
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(id=i, vector=v.tolist(), payload=p)
            for i, (v, p) in enumerate(zip(vectors, payloads))
        ],
        batch_size=256,
        wait=True,
    )
    console.print("[bold green]업로드 완료![/]")

    return model, client

# -----------------------------
# 3) 검색 함수 (필드별 가중치 Rerank)
# -----------------------------
def search_services(query_text: str, model, client, top_k: int = 3):
    """Qdrant 후보(10개) → Tags/서비스명/지원내용 가중치로 rerank → payload 반환"""
    resp = client.query_points(
        collection_name=COLLECTION_NAME,
        query=model.encode(query_text).tolist(),
        limit=10,
    )
    hits = resp.points

    reranked = []
    query_keywords = query_text.split()

    for h in hits:
        payload = h.payload
        bonus = 0.0

        # Tags: +0.5 (정확히 포함 시)
        tags_text = str(payload.get("tags", ""))
        tag_list = [t.strip() for t in re.split(r"[,;\s]\s*", tags_text) if t]
        for kw in query_keywords:
            if kw in tag_list:
                bonus += 0.5

        # 서비스명: +0.2 (부분 포함)
        service_name = str(payload.get("서비스명", ""))
        for kw in query_keywords:
            if kw and kw in service_name:
                bonus += 0.2

        # 지원내용: +0.05 (부분 포함)
        support = str(payload.get("지원내용", ""))
        for kw in query_keywords:
            if kw and kw in support:
                bonus += 0.05

        reranked.append({"hit": h, "final_score": (h.score or 0.0) + bonus})

    # 점수 순 정렬
    reranked.sort(key=lambda x: x["final_score"], reverse=True)

    # 콘솔 표 출력
    table = Table(title=f"'{query_text}' 검색 결과 (Top {top_k})")
    table.add_column("Final", style="cyan")
    table.add_column("Orig", style="dim cyan")
    table.add_column("서비스명", style="magenta")
    table.add_column("Tags", style="green")
    for item in reranked[:top_k]:
        hit = item["hit"]
        payload = hit.payload
        table.add_row(
            f"{item['final_score']:.4f}",
            f"{(hit.score or 0.0):.4f}",
            str(payload.get("서비스명", ""))[:40],
            str(payload.get("tags", ""))[:40],
        )
    console.print(table)

    # 상위 top_k payload만 반환
    return [item["hit"].payload for item in reranked[:top_k]]

# -----------------------------
# 4) Edge-TTS (문장 → MP3 바이트)
# -----------------------------
async def tts_to_mp3_bytes(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    SSML을 쓰지 않는 간단 합성.
    (길이가 길면 자동으로 잘라 읽고 싶으면 아래에서 분할 처리 가능)
    """
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

# (선택) SSML 버전: 항목 사이 잠깐 쉬기
def build_spoken_text(items, pause_ms=400):
    """
    items: payload dict들의 리스트
    반환: TTS로 읽을 자연스러운 한국어 문장
    """
    parts = []
    for i, r in enumerate(items[:MAX_READ], 1):
        title = (r.get("서비스명") or "").strip()
        summary = (r.get("지원내용") or "").strip()
        if len(summary) > SUMMARY_CLIP:
            summary = summary[:SUMMARY_CLIP].rstrip() + " ..."
        # 간단한 문장; SSML 쓰지 않는 텍스트 모드 (edge-tts는 SSML도 지원)
        parts.append(f"{i}번 정책은 {title} 입니다. 요약: {summary}")
    if not parts:
        return "적합한 정책을 찾지 못했습니다. 더 구체적으로 말씀해 주세요."
    return " ".join(parts)

# -----------------------------
# 5) 통합: 검색 → TTS → 파일저장 + base64 반환
# -----------------------------
async def search_and_tts(query: str, model, client, top_k: int = 3, voice: str = DEFAULT_VOICE):
    # 검색
    results = search_services(query, model, client, top_k=top_k)
    # 읽을 문장 생성
    spoken = build_spoken_text(results, pause_ms=400)
    console.print(f"[bold]읽을 문장:[/] {spoken[:120]}{'...' if len(spoken)>120 else ''}")

    # 합성
    mp3_bytes = await tts_to_mp3_bytes(spoken, voice=voice)

    # 저장
    out_file = "policy_tts.mp3"
    with open(out_file, "wb") as f:
        f.write(mp3_bytes)
    console.print(f"[bold green]MP3 저장 완료:[/] {out_file}")

    # 필요 시 base64로도 활용 가능
    b64 = base64.b64encode(mp3_bytes).decode("utf-8")
    return {"results": results, "spoken": spoken, "mp3_b64": b64, "file": out_file}

# -----------------------------
# 6) main
# -----------------------------
if __name__ == "__main__":
    # 1) CSV 로드 & 인덱스 구성
    df, payloads, documents = load_and_prepare(CSV_PATH)
    model, client = build_index(documents, payloads)

    # 2) 쿼리 테스트
    query = "노인 일자리 지원 정책"
    console.print(f"[bold cyan]쿼리:[/] {query}")

    # 3) 검색 → TTS
    asyncio.run(search_and_tts(query, model, client, top_k=3, voice=DEFAULT_VOICE))



