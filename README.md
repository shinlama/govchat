# 🏛️ 정부 정책 검색 챗봇 (GovChat)

AI 기반 정부 복지 정책 검색 및 상담 서비스입니다. 사용자가 자연어로 질문하면 관련 정책을 찾아서 친절하게 설명해드립니다.

## ✨ 주요 기능

- 🔍 **의미 기반 검색**: 자연어 질문으로 정책 검색
- 🤖 **AI 상담원**: GPT를 활용한 친절한 정책 설명
- 📊 **구조화된 데이터**: 정책 조건을 체계적으로 정리
- 🎯 **정확한 매칭**: 벡터 검색으로 관련성 높은 정책 추천

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT-4o-mini, Sentence Transformers
- **Vector DB**: Qdrant
- **Data Processing**: Pandas, SQLite
- **Language**: Python 3.10+

## 📁 프로젝트 구조

```
bockji/
├── app.py                          # Streamlit 챗봇 애플리케이션
├── making_dataset.ipynb            # 데이터 수집 및 정규화 노트북
├── gov24_services.csv             # 수집된 정부 정책 데이터
├── gov24.sqlite                   # SQLite 데이터베이스
├── requirements.txt               # 필요한 패키지 목록
├── making_vectordb_*.ipynb        # 벡터 데이터베이스 생성 노트북
└── README.md                      # 프로젝트 설명서
```

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/shinlama/govchat.git
cd govchat
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. Qdrant 벡터 데이터베이스 실행

```bash
# Docker로 Qdrant 실행
docker run -p 6333:6333 qdrant/qdrant

# 또는 Docker Compose 사용
docker-compose up -d
```

### 5. 애플리케이션 실행

```bash
streamlit run app.py
```

## 🔧 설정

### OpenAI API 키 설정

1. [OpenAI 웹사이트](https://platform.openai.com/api-keys)에서 API 키 발급
2. Streamlit 앱의 사이드바에서 API 키 입력
3. 또는 환경변수로 설정:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### 데이터 업로드

1. Streamlit 앱에서 "CSV 파일 업로드" 섹션으로 이동
2. `gov24_services.csv` 파일 업로드
3. 서비스명 컬럼과 추가 정보 컬럼 선택
4. "Qdrant에 업로드" 버튼 클릭

## 📊 데이터 수집 및 정규화

### 1. 원시 데이터 수집

`making_dataset.ipynb` 노트북을 실행하여 정부24 API에서 데이터를 수집합니다:

```python
# 노트북에서 실행
from making_dataset import collect_to_sqlite

# 전체 데이터 수집
collect_to_sqlite("gov24.sqlite", limit=1000)

# 특정 키워드로 수집
collect_to_sqlite("gov24.sqlite", keyword="청년", limit=100)
```

### 2. AI 기반 데이터 정규화

OpenAI API를 사용하여 서술형 조건을 구조화된 데이터로 변환:

```python
# 나이 조건 정규화
age_data = normalize_conditions_with_openai("만 18세 이상 65세 미만", "age")
# 결과: {"min_age": 18, "max_age": 65, "age_unit": "세"}

# 소득 조건 정규화
income_data = normalize_conditions_with_openai("중위소득 150% 이하", "income")
# 결과: {"income_type": "중위소득", "income_percentage": 150}
```

## 🎯 사용 예시

### 검색 질문 예시

- "청년 주거 지원 받을 수 있는 제도 알려줘"
- "소득이 적은 가정을 위한 지원 정책이 뭐가 있어?"
- "장애인을 위한 취업 지원 프로그램 찾아줘"
- "농업인 대상 정책들 보여줘"

### 응답 예시

```
🤖 챗봇 답변

다음과 같은 청년 주거 지원 정책들을 찾았습니다:

## 1. 청년 주거안정 월세대출
- **지원내용**: 월세 자금 대출 및 보증
- **대상**: 만 35세 이하 청년
- **조건**: 보증금 1억원 이하, 월세 60만원 이하
- **신청방법**: 주택금융공사 온라인 신청

## 2. 청년 전세자금 대출
- **지원내용**: 전세보증금 대출
- **대상**: 만 39세 이하 무주택 청년
- **조건**: 중위소득 150% 이하
- **신청방법**: 주택금융공사 지점 방문
```

## 📈 데이터 구조

### 정규화된 컬럼들

- **기본 정보**: `service_id`, `title`, `category`, `org_name`
- **나이 조건**: `age_min`, `age_max`, `age_unit`, `age_restriction`
- **소득 조건**: `income_type`, `income_percentage`, `income_amount`
- **지원대상**: `target_age_group`, `target_employment`, `target_special`
- **선정기준**: `selection_method`, `required_documents`, `evaluation_criteria`
- **거주지 조건**: `residence_type`, `specific_regions`, `residence_duration`

## 🔍 검색 기능

### 의미 기반 검색

- **한국어 임베딩**: `jhgan/ko-sroberta-multitask` 모델 사용
- **벡터 검색**: Qdrant를 통한 고속 유사도 검색
- **BM25 정렬**: 검색 결과의 관련성 순 정렬

### 검색 옵션

- **검색 결과 개수**: 1-10개 조정 가능
- **실시간 검색**: 입력과 동시에 결과 표시
- **다중 컬럼 검색**: 제목, 내용, 기관명 등 종합 검색

## 🛡️ 보안 및 개인정보

- API 키는 환경변수로 관리
- 개인정보는 수집하지 않음
- 공개 데이터만 사용

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

- **프로젝트 링크**: [https://github.com/shinlama/govchat](https://github.com/shinlama/govchat)

## 🙏 
- [정부24 API](https://www.gov.kr/portal/api) - 정책 데이터 제공
- [OpenAI](https://openai.com/) - AI 모델 제공
- [Qdrant](https://qdrant.tech/) - 벡터 데이터베이스
- [Streamlit](https://streamlit.io/) - 웹 애플리케이션 프레임워크

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!
