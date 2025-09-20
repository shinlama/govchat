import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import uuid
import tempfile
import wave
import numpy as np
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import av

# -----------------------------
# 초기 세팅
# -----------------------------
st.set_page_config(page_title="Vector Search Chatbot with Voice", layout="wide")
st.title("💬 음성 입력 정책 서비스 검색 챗봇")

# OpenAI API Key 입력
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

if openai_api_key:
    st.session_state.openai_api_key = openai_api_key

# Qdrant 설정
st.sidebar.header("Qdrant 설정")
qdrant_host = st.sidebar.text_input("Qdrant Host", "localhost")
qdrant_port = st.sidebar.number_input("Qdrant Port", value=6333)
collection_name = st.sidebar.text_input("Collection Name", "demo_collection")

# 모델 로딩
@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

embedder = load_model()

# Whisper 모델 로딩 (CPU 전용, int8 최적화)
@st.cache_resource
def load_stt_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

stt_model = load_stt_model()

# Qdrant client 연결
qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

# OpenAI 클라이언트
def get_openai_client():
    if st.session_state.openai_api_key:
        return OpenAI(api_key=st.session_state.openai_api_key)
    return None


# -----------------------------
# CSV 업로드
# -----------------------------
st.header("📂 데이터 업로드")
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:")
    st.dataframe(df.head())

    text_columns = df.columns.tolist()
    main_col = st.selectbox("서비스명을 나타내는 컬럼 선택", text_columns)
    support_cols = st.multiselect("추가 정보를 담은 컬럼 선택", text_columns, default=[c for c in text_columns if c != main_col])

    if st.button("Qdrant에 업로드"):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedder.get_sentence_embedding_dimension(),
                                        distance=Distance.COSINE)
        )
        points = []
        for _, row in df.iterrows():
            concat_text = str(row[main_col])
            for c in support_cols:
                concat_text += " " + str(row[c])

            vector = embedder.encode(concat_text).tolist()
            payload = {col: str(row[col]) for col in df.columns}
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        qdrant_client.upsert(collection_name=collection_name, points=points)
        st.success(f"✅ {len(points)}개 항목을 Qdrant에 업로드 완료!")


# -----------------------------
# 음성 입력 → 텍스트 변환
# -----------------------------
st.header("🎙 음성 입력")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_frames.append(audio)
        return frame

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"audio": True, "video": False},
)

query_text = st.text_input("✍️ 직접 텍스트 입력", placeholder="예: 청년 주거 지원 받을 수 있는 제도 알려줘")

if ctx.audio_processor:
    if st.button("🎙 음성 변환 → 텍스트 입력"):
        pcm_data = np.concatenate(ctx.audio_processor.audio_frames, axis=1).T
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp_wav.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_data.astype(np.int16).tobytes())

        segments, _ = stt_model.transcribe(tmp_wav.name, beam_size=5)
        recognized_text = " ".join([seg.text for seg in segments])

        st.session_state["query_text"] = recognized_text
        st.success("📝 인식된 텍스트: " + recognized_text)

# -----------------------------
# 의미 기반 검색 + GPT 요약
# -----------------------------
st.header("🔍 의미 기반 검색 & 요약 답변")

query = st.session_state.get("query_text", query_text)
top_k = st.slider("검색 결과 개수", 1, 10, 5)

if st.button("검색 실행"):
    if not query:
        st.warning("검색어를 입력해주세요.")
    else:
        q_emb = embedder.encode(query).tolist()
        results = qdrant_client.search(collection_name=collection_name, query_vector=q_emb, limit=top_k)

        if not results:
            st.error("검색 결과가 없습니다.")
        else:
            context_texts = []
            for r in results:
                p = r.payload
                main_text = p.get(main_col, "")
                extras = [f"{col}: {p.get(col, '')}" for col in support_cols]
                context_texts.append(f"- {main_text}\n  " + "\n  ".join(extras))

            context = "\n\n".join(context_texts)
            prompt = f"""
당신은 정부 복지/정책 서비스 안내 챗봇입니다.
아래는 데이터베이스에서 검색된 후보 정책들입니다. 
사용자의 질문에 맞게 적절히 요약하여 답변해 주세요. 

사용자 질문: {query}

검색된 후보 정책:
{context}

요약 규칙:
- 서비스명은 제목처럼 강조
- 지원내용, 신청기한 등은 핵심만 정리
- 사용자 질문에 맞는 서비스만 선택해 친절하게 설명
- 리스트 형식으로 답변

최종 답변:
"""

            openai_client = get_openai_client()
            
            if openai_client:
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "너는 정부 지원 정책을 요약해서 알려주는 친절한 상담원이다."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 🤖 챗봇 답변")
                    st.write(answer)
                except Exception as e:
                    st.error(f"OpenAI API 오류: {e}")
            else:
                st.error("⚠️ OpenAI API Key를 입력해주세요.")
