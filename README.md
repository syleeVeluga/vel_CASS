# CASS Lite — 범죄분석 선별 시스템 (경량화 버전)

> PDF 조서를 업로드하면 AI가 혐의점과 모순을 찾아주는 로컬 수사 보조 도구

## 주요 기능

1. **PDF 조서 파싱** — docling으로 텍스트 추출 후 정규식으로 문답(Q&A) 구조화
2. **데이터 수정** — 파싱 결과를 엑셀처럼 편집 가능 (st.data_editor)
3. **AI 2단계 검증** — Analyst → Critic으로 할루시네이션 방지
4. **체크리스트 통합 리포트** — 행동분석/위협평가 4대 영역 자동 판정

## 지원 모델

| Provider | 모델 | Reasoning |
|---|---|---|
| OpenAI | GPT-5.2, GPT-5.2 Pro | reasoning_effort |
| Google | Gemini 3 Flash, Gemini 3 Pro | thinking_level |

## 설치

```bash
# Python 3.10+ 필요
pip install -r requirements.txt
```

## 사용법

```bash
# 1. API Key 설정 (.env 파일)
cp .env.example .env
# .env 파일에 API Key 입력

# 2. 실행
streamlit run app.py
```

## 프로젝트 구조

```
├── app.py                  # Streamlit 메인 앱
├── parsing/
│   └── pdf_parser.py       # PDF 파싱 + 정규식 Q&A 변환
├── analysis/
│   ├── chunker.py          # 스마트 청킹 (20 Q&A + 3 오버랩)
│   ├── prompts.py          # 시스템 프롬프트 (영어)
│   └── llm_utils.py        # LLM API 호출 (GPT-5.2 + Gemini 3)
├── output/                 # CSV 출력
├── tests/                  # 단위 테스트
├── requirements.txt
└── .env.example
```

## 검증 시나리오

1. **파싱 수정 반영** — data_editor에서 수정 후 분석 시 반영 확인
2. **할루시네이션 차단** — Critic 단계에서 기각된 건수 로그 확인
3. **CSV 저장** — output/parsed_current.csv 생성 확인

## 🚀 배포 (Deployment)

이 프로젝트는 **Streamlit Community Cloud**를 통해 무료로 쉽게 배포할 수 있습니다. (GitHub Pages는 정적 사이트만 지원하므로 Python 앱 구동이 불가능합니다.)

### 1단계: Streamlit Cloud 연결
1. **[Streamlit Community Cloud](https://share.streamlit.io/)** 접속 및 로그인 (GitHub 계정 연동)
2. **New app** 버튼 클릭
3. **Repository**: `syleeVeluga/vel_CASS` 선택
4. **Main file path**: `app.py` 입력
5. **Deploy!** 버튼 클릭

### 2단계: API Key 설정
배포가 완료되면 앱이 실행되지만 API Key가 없어서 분석 기능이 작동하지 않습니다.
1. 배포된 앱 우측 하단 **Manage app** 클릭
2. **Settings** (점 3개) > **Secrets** 메뉴 이동
3. 아래 내용을 복사하여 붙여넣고 저장합니다:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GOOGLE_API_KEY = "AIza..."
   ```
4. 앱을 재부팅(Reboot)하면 정상 작동합니다.

### CI/CD (GitHub Actions)
`.github/workflows/ci.yml`이 설정되어 있어, 코드가 푸시될 때마다 자동으로 단위 테스트(`pytest`)가 실행되어 품질을 검증합니다.
