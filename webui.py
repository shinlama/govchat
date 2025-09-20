# webui.py
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
API_BASE = st.sidebar.text_input("API Base URL", "http://165.132.46.88:30984")
ENGINE   = st.sidebar.selectbox("STT 엔진", ["fw", "ow"], index=0)
LANG     = st.sidebar.text_input("언어", "ko")
VOICE    = st.sidebar.selectbox("TTS 음성", ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"], index=0)
TOPK     = st.sidebar.number_input("검색 TopK", min_value=1, max_value=10, value=3)
BEAM     = st.sidebar.number_input("Faster-Whisper beam_size", min_value=1, max_value=10, value=5)
TIMEOUT  = st.sidebar.number_input("요청 타임아웃(sec)", min_value=5, max_value=300, value=120)

PIPELINE_URL   = f"{API_BASE}/stt_search_tts"
HEALTHZ_URL    = f"{API_BASE}/healthz"

st.caption("TIP: 먼저 백엔드 서버를 켜세요 → `uvicorn app.server:app --port 30984 --reload`")

# -----------------------------
# WebRTC 오디오 수집 (마이크)
# -----------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffers = []
        self.sr = 48000

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # float32 PCM, shape = (channels, samples)
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
    )

    c1, c2 = st.columns(2)
    with c1:
        if ctx and ctx.state.playing and st.button("🎧 현재 녹음분 전송"):
            try:
                path = save_wav_from_buffers(ctx.audio_receiver.get_frames(), sr=48000)
                if not path:
                    st.warning("수집된 오디오가 없습니다.")
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
                        t0 = time.time()
                        res = requests.post(PIPELINE_URL, files=files, data=data, timeout=TIMEOUT)
                        dt = time.time() - t0
                    if res.ok:
                        st.session_state.last_json = res.json()
                        st.success(f"성공! (RTT {dt:.2f}s)")
                    else:
                        st.error(f"오류: {res.status_code} {res.text}")
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
            files = {"audio": (up.name, up, "application/octet-stream")}
            data = {
                "engine": ENGINE,
                "language": LANG,
                "beam_size": int(BEAM),
                "topk": int(TOPK),
                "voice": VOICE,
            }
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

    st.markdown("### 📝 STT 결과")
    st.write(js.get("stt", {}))

    st.markdown("### 🔎 검색 결과")
    st.write(js.get("search", {}))

    st.markdown("### 🔊 합성 음성 (TTS)")
    tts = js.get("tts", {})
    b64 = tts.get("audio_mp3_b64")
    if b64:
        try:
            st.audio(base64.b64decode(b64), format="audio/mp3")
        except Exception as e:
            st.error(f"오디오 디코딩 오류: {e}")
    else:
        st.info("오디오 데이터가 없습니다.")

    with st.expander("읽어준 문장 확인"):
        st.write(tts.get("spoken_text", ""))
else:
    st.info("아직 결과가 없습니다. 위 탭에서 마이크 녹음 또는 파일 업로드 후 전송하세요.")
