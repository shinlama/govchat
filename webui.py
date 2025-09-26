import io
import base64
import time
import requests
import numpy as np
import streamlit as st

# WebRTC로 마이크 녹음
import av
from scipy.io.wavfile import write
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

st.set_page_config(page_title="통합 STT→정책검색→TTS", layout="centered")
st.title("음성 복지정책 도우미 (통합 서버 테스트 UI)")

# -----------------------------
# 사이드바: 서버/옵션
# -----------------------------
st.sidebar.header("서버 & 옵션")
# 배포된 서버 주소로 기본값 설정
API_BASE = st.sidebar.text_input("API Base URL", "http://165.132.46.88:32374") 
ENGINE = st.sidebar.selectbox("STT 엔진", ["fw", "ow"], index=0)
LANG = st.sidebar.text_input("언어", "ko")
VOICE = st.sidebar.selectbox("TTS 음성", ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"], index=0)
TOPK = st.sidebar.number_input("검색 TopK", min_value=1, max_value=10, value=3)
BEAM = st.sidebar.number_input("Faster-Whisper beam_size", min_value=1, max_value=10, value=5)
TIMEOUT = st.sidebar.number_input("요청 타임아웃(sec)", min_value=5, max_value=300, value=120)

# STT, 검색, TTS 통합 파이프라인 엔드포인트 사용
PIPELINE_URL = f"{API_BASE}/stt_search_tts"
HEALTHZ_URL = f"{API_BASE}/healthz"

st.caption("TIP: 백엔드 서버는 `http://165.132.46.88:32374`에 **/stt_search_tts** 엔드포인트가 배포되어 있어야 합니다.")

# -----------------------------
# WebRTC 오디오 수집 (마이크)
# -----------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffers = []
        self.sr = 48000
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.buffers.append(frame.to_ndarray())
        return frame

def save_wav_from_buffers(buffers, sr=48000, path="tmp_input.wav"):
    if not buffers:
        return None
    data = np.concatenate(buffers, axis=1)
    if data.ndim == 2 and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)  # stereo -> mono
    data = (data.squeeze() * 32767).astype("int16")
    write(path, sr, data)
    return path

tabs = st.tabs(["🎙️ 마이크 녹음", "📁 파일 업로드"])

# 상태 저장
if "last_json" not in st.session_state:
    st.session_state.last_json = None

# -----------------------------
# 탭 1: 마이크 녹음
# -----------------------------
with tabs[0]:
    st.subheader("🎙️ 마이크 → /stt_search_tts")
    st.markdown("1) **Start** 눌러 말하고 → 2) **🎧 현재 녹음분 전송**")

    ctx = webrtc_streamer(
        key="stt-pipeline",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
        audio_processor_factory=AudioProcessor,
    )

    c1, c2 = st.columns(2)
    with c1:
        if ctx and ctx.state.playing and st.button("🎧 현재 녹음분 전송"):
            try:
                if ctx.audio_processor:
                    path = save_wav_from_buffers(ctx.audio_processor.buffers, sr=48000)
                    if not path:
                        st.warning("수집된 오디오가 없습니다. 마이크 녹음 상태를 확인해주세요.")
                    else:
                        with open(path, "rb") as f:
                            files = {"audio": ("input.wav", f, "audio/wav")}
                            data = {
                                "engine": ENGINE,
                                "language": LANG,
                                "beam_size": int(BEAM),
                                "topk": int(TOPK),
                                "voice": VOICE,
                            }
                            st.spinner("서버에 요청 중...")
                            t0 = time.time()
                            res = requests.post(PIPELINE_URL, files=files, data=data, timeout=TIMEOUT)
                            dt = time.time() - t0
                        if res.ok:
                            st.session_state.last_json = res.json()
                            st.success(f"성공! (RTT {dt:.2f}s)")
                        else:
                            st.error(f"오류: {res.status_code} {res.text}")
                else:
                    st.error("오디오 프로세서가 준비되지 않았습니다.")
            except Exception as e:
                st.error(f"전송 중 오류: {e}")

    with c2:
        if st.button("🧪 서버 상태 체크(/healthz)"):
            try:
                hres = requests.get(HEALTHZ_URL, timeout=10)
                st.write(hres.json() if hres.ok else hres.text)
            except Exception as e:
                st.error(f"healthz 실패: {e}")

# -----------------------------
# 탭 2: 파일 업로드
# -----------------------------
with tabs[1]:
    st.subheader("📁 파일 → /stt_search_tts")
    up = st.file_uploader("오디오 파일 업로드 (wav/mp3/m4a 등)", type=["wav", "mp3", "m4a"])
    if up and st.button("🚀 업로드 파일로 요청 보내기"):
        try:
            audio_bytes = up.read()
            files = {"audio": (up.name, audio_bytes, up.type)}
            data = {
                "engine": ENGINE,
                "language": LANG,
                "beam_size": int(BEAM),
                "topk": int(TOPK),
                "voice": VOICE,
            }
            st.spinner("서버에 요청 중...")
            t0 = time.time()
            res = requests.post(PIPELINE_URL, files=files, data=data, timeout=TIMEOUT)
            dt = time.time() - t0
            if res.ok:
                st.session_state.last_json = res.json()
                st.success(f"성공! (RTT {dt:.2f}s)")
            else:
                st.error(f"오류: {res.status_code} {res.text}")
        except Exception as e:
            st.error(f"요청 실패: {e}")

# -----------------------------
# 결과 표시/재생
# -----------------------------
st.divider()
st.subheader("결과")

if st.session_state.last_json:
    js = st.session_state.last_json

    # STT 결과
    st.markdown("### 📝 STT 결과")
    st.write(js.get("stt", {}))

    # 검색 결과
    st.markdown("### 🔎 검색 결과")
    # 서버 응답 구조: js['search']['results']
    results = js.get("search", {}).get("results", []) 
    if results:
        cols = st.columns(3)
        for i, item in enumerate(results):
            # 검색 결과에서 필요한 필드를 안전하게 추출
            service_name = item.get('service_name') or item.get('서비스명', 'N/A')
            support_content = item.get('support') or item.get('지원내용', 'N/A')
            tags = item.get('tags', 'N/A')
            
            # 태그가 리스트 형태일 경우 보기 좋게 변환
            if isinstance(tags, list):
                tags_text = ", ".join(t.strip("['\"") for t in tags)
            else:
                tags_text = tags.strip("['\"")

            with cols[i % 3]:
                # 서비스명과 랭크를 제목으로 사용
                title_text = f"{i+1}. {service_name}"
                with st.expander(f"**{title_text}**", expanded=True):
                    # 지원내용과 태그가 성공적으로 표시됨
                    st.markdown(f"**지원내용:** {support_content}")
                    st.markdown(f"**태그:** {tags_text}")
    else:
        st.info("검색 결과가 없습니다.")

    # 합성 음성
    st.markdown("### 🔊 합성 음성 (TTS)")
    tts = js.get("tts", {})
    b64 = tts.get("audio_mp3_b64")
    
    # TTS 결과에서 'spoken_text' 필드를 추출하여 읽어줄 문장 확인
    spoken_text = tts.get("spoken_text") or js.get("summary", "읽어줄 문장이 없습니다.")

    if b64:
        # Base64 데이터가 있을 경우에만 재생 시도
        try:
            st.audio(base64.b64decode(b64), format="audio/mp3")
        except Exception as e:
            st.error(f"오디오 디코딩 오류: {e}")
    else:
        # 오디오 바이트가 비어있을 경우, 서버 측 TTS 오류를 의심
        st.error("오디오 데이터가 서버에서 생성되지 않았습니다. (서버 측 TTS 오류 가능성)")

    with st.expander("읽어준 문장 확인"):
        # spoken_text 필드를 출력
        st.write(spoken_text)
else:
    st.info("아직 결과가 없습니다. 위 탭에서 마이크 녹음 또는 파일 업로드 후 전송하세요.")
