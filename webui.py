import io
import base64
import time
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# streamlit-audio-recorder로 마이크 녹음
from streamlit_mic_recorder import mic_recorder
import tempfile
import os

st.set_page_config(page_title="통합 STT→정책검색→TTS", layout="centered")
st.title("👩🏻‍💼 음성 복지정책 도우미")

# -----------------------------
# 사이드바: 서버/옵션
# -----------------------------
st.sidebar.header("서버 & 옵션")
# 배포된 서버 주소로 기본값 설정
API_BASE = st.sidebar.text_input("API Base URL", "http://165.132.46.88:31180") 
ENGINE = st.sidebar.selectbox("STT 엔진", ["fw", "ow"], index=0)
LANG = st.sidebar.text_input("언어", "ko")
VOICE = st.sidebar.selectbox("TTS 음성", ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"], index=0)
TOPK = st.sidebar.number_input("검색 TopK", min_value=1, max_value=10, value=3)
BEAM = st.sidebar.number_input("Faster-Whisper beam_size", min_value=1, max_value=10, value=5)
TIMEOUT = st.sidebar.number_input("요청 타임아웃(sec)", min_value=5, max_value=300, value=120)

# OpenAI API Key 설정
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
# STT, 검색, TTS 통합 파이프라인 엔드포인트 사용
PIPELINE_URL = f"{API_BASE}/stt_search_tts"

# -----------------------------
# OpenAI를 사용한 자연스러운 문장 생성 함수
# -----------------------------
def generate_policy_summary(service_data):
    """정책 정보를 핵심 요약으로 변환"""
    if not OPENAI_API_KEY:
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""
다음 정책 정보를 핵심 내용만 요약해서 설명해주세요:

서비스명: {service_data.get('service_name', 'N/A')}
지원내용: {service_data.get('support', 'N/A')}
신청대상: {service_data.get('target_beneficiaries', 'N/A')}
신청기간: {service_data.get('application_deadline', 'N/A')}
신청방법: {service_data.get('application_method', 'N/A')}
문의처: {service_data.get('contact', 'N/A')}
필요서류: {service_data.get('required_documents', 'N/A')}

요구사항:
1. 어떤 정책인지 (지원내용)
2. 신청 대상
3. 신청 기간
4. 신청 방법
5. 필요한 서류
6. 문의처
이 6가지 핵심 정보를 간결하게 정리해서 설명

핵심 요약:
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 정부 정책의 핵심 정보를 명확하게 정리하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"정책 요약 생성 중 오류: {e}")
        return None

def generate_field_summary(service_data, field_name):
    """특정 필드의 내용을 요약"""
    if not OPENAI_API_KEY:
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        field_value = service_data.get(field_name, 'N/A')
        if field_value == 'N/A' or not field_value:
            return None
        
        field_labels = {
            'support': '지원내용',
            'target_beneficiaries': '신청대상',
            'application_deadline': '신청기간',
            'application_method': '신청방법',
            'required_documents': '필요서류',
            'contact': '문의처'
        }
        
        field_label = field_labels.get(field_name, field_name)
        
        prompt = f"""
다음 정보를 간결하고 이해하기 쉽게 요약해주세요:

{field_value}

요구사항:
1. 핵심 내용만 간결하게 정리
2. 이해하기 쉬운 문장으로 작성
3. 불필요한 반복 제거
4. 한국어로 작성

요약:
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"당신은 {field_label} 정보를 간결하게 요약하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

def generate_tts_summary(service_data):
    """TTS용 4줄 요약 생성"""
    if not OPENAI_API_KEY:
        return None
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""
다음 정책 정보를 4줄 이내의 자연스러운 문장으로 요약해주세요:

서비스명: {service_data.get('service_name', 'N/A')}
지원내용: {service_data.get('support', 'N/A')}
신청대상: {service_data.get('target_beneficiaries', 'N/A')}
신청방법: {service_data.get('application_method', 'N/A')}
필요서류: {service_data.get('required_documents', 'N/A')}
문의처: {service_data.get('contact', 'N/A')}

요구사항:
0. []안에는 정책 정보 채워넣기
1. "추천하는 정책은 [정책명]입니다."로 시작
2. "대상은 [신청대상]이며"로 이어짐
3. "신청 방법은 [신청방법]이고"로 이어짐
4. "[필요서류]를 통해 [신청방법]으로 신청하면 됩니다. 문의처는 [문의처]입니다."로 마무리
5. 4줄 이내의 자연스러운 문장으로 작성
6. 음성으로 읽기 좋게 작성

요약:
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 정책 정보를 음성으로 읽기 좋은 자연스러운 문장으로 요약하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

def display_policy_info(service_data, index):
    """정책 정보를 Streamlit UI 스타일로 표시"""
    service_name = service_data.get('service_name', 'N/A')
    
    # 카드 형태로 표시
    with st.container():
        st.markdown(f"### {index+1}. {service_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 지원내용**")
            if OPENAI_API_KEY:
                with st.spinner("지원내용 요약 중..."):
                    support_summary = generate_field_summary(service_data, 'support')
                    st.write(support_summary if support_summary else service_data.get('support', 'N/A'))
            else:
                st.write(service_data.get('support', 'N/A'))
            
            st.markdown("**👥 신청대상**")
            if OPENAI_API_KEY:
                with st.spinner("신청대상 요약 중..."):
                    target_summary = generate_field_summary(service_data, 'target_beneficiaries')
                    st.write(target_summary if target_summary else service_data.get('target_beneficiaries', 'N/A'))
            else:
                st.write(service_data.get('target_beneficiaries', 'N/A'))
            
            st.markdown("**📅 신청기간**")
            if OPENAI_API_KEY:
                with st.spinner("신청기간 요약 중..."):
                    deadline_summary = generate_field_summary(service_data, 'application_deadline')
                    st.write(deadline_summary if deadline_summary else service_data.get('application_deadline', 'N/A'))
            else:
                st.write(service_data.get('application_deadline', 'N/A'))
        
        with col2:
            st.markdown("**📝 신청방법**")
            if OPENAI_API_KEY:
                with st.spinner("신청방법 요약 중..."):
                    method_summary = generate_field_summary(service_data, 'application_method')
                    st.write(method_summary if method_summary else service_data.get('application_method', 'N/A'))
            else:
                st.write(service_data.get('application_method', 'N/A'))
            
            st.markdown("**📄 필요서류**")
            if OPENAI_API_KEY:
                with st.spinner("필요서류 요약 중..."):
                    docs_summary = generate_field_summary(service_data, 'required_documents')
                    st.write(docs_summary if docs_summary else service_data.get('required_documents', 'N/A'))
            else:
                st.write(service_data.get('required_documents', 'N/A'))
            
            st.markdown("**📞 문의처**")
            if OPENAI_API_KEY:
                with st.spinner("문의처 요약 중..."):
                    contact_summary = generate_field_summary(service_data, 'contact')
                    st.write(contact_summary if contact_summary else service_data.get('contact', 'N/A'))
            else:
                st.write(service_data.get('contact', 'N/A'))
        
        st.markdown("---")

# -----------------------------
# streamlit-audio-recorder 기반 오디오 수집
# -----------------------------
def save_audio_to_temp_file(audio_data, filename="tmp_input.wav"):
    """오디오 데이터를 임시 파일로 저장"""
    if audio_data is None:
        return None
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_data)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        st.error(f"오디오 파일 저장 중 오류: {e}")
        return None

# 상태 저장
if "last_json" not in st.session_state:
    st.session_state.last_json = None
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None

# -----------------------------
# 마이크 녹음 (streamlit-audio-recorder 사용)
# -----------------------------
st.subheader("상황에 맞는 복지 정책을 찾아 드려요!")
st.markdown("**Start Recording 버튼**을 클릭하여 말하고 **Stop 버튼**으로 중단하면 자동 검색됩니다!")

# streamlit-audio-recorder 사용
wav_audio_data = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop",
    just_once=True,
    use_container_width=True,
    key="recorder"
)

# 녹음된 오디오가 있으면 자동으로 서버에 전송
if wav_audio_data and "bytes" in wav_audio_data:
    if st.session_state.recorded_audio != wav_audio_data["bytes"]:
        st.session_state.recorded_audio = wav_audio_data["bytes"]
        
        # 자동으로 서버에 전송
        try:
            # 임시 파일로 저장
            path = save_audio_to_temp_file(st.session_state.recorded_audio)
            if not path:
                st.warning("오디오 파일 생성에 실패했습니다.")
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
                    
                    # 1단계: 검색 결과 받기
                    with st.spinner("서버에 요청 중..."):
                        t0 = time.time()
                        res = requests.post(PIPELINE_URL, files=files, data=data, timeout=TIMEOUT)
                        dt = time.time() - t0
                    
                    # 2단계: GPT 요약 생성 후 TTS 요청
                    if res.ok and OPENAI_API_KEY:
                        search_results = res.json().get("search", {}).get("results", [])
                        if search_results:
                            with st.spinner("GPT 요약 생성 중..."):
                                tts_summary = generate_tts_summary(search_results[0])
                                if tts_summary:
                                    # GPT 요약 텍스트로 /synthesize 엔드포인트에 TTS 요청
                                    synthesize_url = f"{API_BASE}/synthesize"
                                    tts_data = {
                                        "text": tts_summary,
                                        "voice": VOICE,
                                        "rate": "+0%",
                                        "volume": "+0%",
                                        "pitch": "+0Hz"
                                    }
                                    
                                    with st.spinner("GPT 요약으로 음성 생성 중..."):
                                        t0 = time.time()
                                        tts_res = requests.post(synthesize_url, json=tts_data, timeout=TIMEOUT)
                                        dt = time.time() - t0
                                        
                                        if tts_res.ok:
                                            # TTS 결과를 기존 응답에 병합
                                            tts_result = tts_res.json()
                                            response_data = res.json()
                                            if "tts" not in response_data:
                                                response_data["tts"] = {}
                                            response_data["tts"]["audio_mp3_b64"] = tts_result.get("mp3_b64", "")
                                            response_data["tts"]["spoken_text"] = tts_summary
                                            response_data["tts"]["voice"] = VOICE
                                            response_data["tts"]["synthesis_s"] = dt
                                            
                                            # 세션에 병합된 결과 저장
                                            st.session_state.last_json = response_data
                                        else:
                                            st.warning(f"TTS 생성 실패: {tts_res.status_code}")
                                            # 기존 결과 그대로 사용
                                            st.session_state.last_json = res.json()
                    
                    if res.ok:
                        # GPT TTS가 처리되지 않은 경우에만 기본 결과 저장
                        if not (OPENAI_API_KEY and res.json().get("search", {}).get("results", [])):
                            st.session_state.last_json = res.json()
                        
                        # 임시 파일 정리
                        try:
                            os.remove(path)
                        except:
                            pass
                    else:
                        st.error(f"오류: {res.status_code} {res.text}")
                        
        except Exception as e:
            st.error(f"전송 중 오류: {e}")

# -----------------------------
# 결과 표시/재생
# -----------------------------
st.divider()

if st.session_state.last_json:
    js = st.session_state.last_json

    # STT 결과
    st.markdown("### 📝 인식된 음성")
    stt_data = js.get("stt", {})
    recognized_text = stt_data.get("text", "음성을 인식하지 못했습니다.")
    
    # 인식된 텍스트를 깔끔하게 표시
    if recognized_text and recognized_text.strip():
        st.markdown(f'> "{recognized_text}"')
    else:
        st.info("음성을 인식하지 못했습니다.")

    # 검색 결과
    st.markdown("### 🔎 지원 정책 검색 결과")
    # 서버 응답 구조: js['search']['results'] 
    results = js.get("search", {}).get("results", []) 
    if results:
        for i, item in enumerate(results):
            # 검색 결과에서 필요한 필드를 안전하게 추출
            service_name = item.get('service_name') or item.get('서비스명', 'N/A')
            support_content = item.get('support') or item.get('지원내용', 'N/A')
            target_beneficiaries = item.get('target_beneficiaries', 'N/A')
            application_deadline = item.get('application_deadline', 'N/A')
            application_method = item.get('application_method', 'N/A')
            contact = item.get('contact', 'N/A')
            required_documents = item.get('required_documents', 'N/A')

            # 새로운 함수를 사용하여 정책 정보 표시
            display_policy_info(item, i)
    else:
        st.info("검색 결과가 없습니다.")

    # 합성 음성
    st.markdown("### 🔊 음성 지원")
    tts = js.get("tts", {})
    # 서버별 키 호환: audio_mp3_b64 또는 mp3_b64
    b64 = tts.get("audio_mp3_b64") or tts.get("mp3_b64")

    # GPT API로 4줄 요약 생성
    if results and OPENAI_API_KEY:
        with st.spinner("음성용 정책 요약 생성 중..."):
            tts_summary = generate_tts_summary(results[0])  # 첫 번째 결과 사용
            spoken_text = tts_summary if tts_summary else tts.get("spoken_text") or js.get("summary", "읽어줄 문장이 없습니다.")
    else:
        spoken_text = tts.get("spoken_text") or js.get("summary", "읽어줄 문장이 없습니다.")

    def safe_b64_decode(s: str) -> bytes:
        if not isinstance(s, str):
            raise ValueError("base64 문자열이 아님")
        clean = s.strip().replace("\n", "").replace("\r", "")
        # 공백이 '+'로 잘려온 경우 보정
        clean = clean.replace(" ", "+")
        # 패딩 보정
        pad = (-len(clean)) % 4
        if pad:
            clean += "=" * pad
        return base64.b64decode(clean)

    if b64:
        try:
            audio_bytes = safe_b64_decode(b64)
            # Base64 인코딩된 오디오를 HTML audio 태그로 자동 재생
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"오디오 디코딩 오류: {e}")
    else:
        st.error("오디오 데이터가 서버에서 생성되지 않았습니다. (서버 측 TTS 키(audio_mp3_b64/mp3_b64) 확인 필요)")
else:
    st.info("아직 결과가 없습니다. 위 탭에서 마이크 녹음 또는 파일 업로드 후 전송하세요.")
