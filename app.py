"""
CASS Lite — 범죄분석 선별 시스템 (경량화 버전)
PDF 조서 → 문답 파싱 → AI 2단계 검증 → 체크리스트 통합 리포트
"""

import json
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────
# PDF 생성을 위한 라이브러리
# ─────────────────────────────────────────────
import markdown
from xhtml2pdf import pisa
from io import BytesIO

from parsing.pdf_parser import extract_text, parse_qa
from analysis.chunker import create_chunks
from analysis.llm_utils import (
    LLMConfig,
    AVAILABLE_MODELS,
    REASONING_LEVELS,
    call_analyst,
    call_critic,
    call_reporter,
)

# .env 파일 경로 (app.py 기준)
_ENV_PATH = Path(__file__).parent / ".env"

# KST (Korea Standard Time)
KST = timezone(timedelta(hours=9))


def init_page():
    """페이지 기본 설정."""
    st.set_page_config(
        page_title="CASS Lite — 범죄분석 선별 시스템",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def setup_sidebar() -> Optional[LLMConfig]:
    """사이드바 설정."""
    st.sidebar.title("⚙️ AI 설정")
    st.sidebar.divider()

    provider = st.sidebar.selectbox(
        "LLM 프로바이더",
        options=list(AVAILABLE_MODELS.keys()),
        help="분석에 사용할 AI 모델 프로바이더를 선택하세요.",
    )

    model_options = AVAILABLE_MODELS[provider]
    model_label = st.sidebar.selectbox(
        "모델",
        options=list(model_options.keys()),
    )
    model_id = model_options[model_label]

    if provider == "OpenAI":
        reasoning_options = REASONING_LEVELS["OpenAI"]
    else:
        reasoning_options = REASONING_LEVELS["Gemini"].get(model_id, ["low", "high"])

    reasoning_level = st.sidebar.select_slider(
        "Reasoning 레벨",
        options=reasoning_options,
        value=reasoning_options[len(reasoning_options) // 2],
        help="높을수록 깊이 사고하지만 응답이 느려질 수 있습니다.",
    )

    st.sidebar.divider()

    # API Key 로드
    load_dotenv(_ENV_PATH, override=True)
    env_key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GOOGLE_API_KEY"

    env_values = dotenv_values(_ENV_PATH)
    env_key = (env_values.get(env_key_name) or "").strip()
    if not env_key:
        env_key = (os.getenv(env_key_name) or "").strip()
    has_env_key = bool(env_key)

    api_key = ""

    if has_env_key:
        st.sidebar.success(
            f"🔐 **{env_key_name}** — .env에서 로드됨\n\n키가 안전하게 설정되어 있습니다.",
            icon="✅",
        )
        api_key = env_key
    else:
        st.sidebar.info(
            f"📝 .env에 `{env_key_name}`이 없습니다.\n아래에서 직접 입력하세요.",
            icon="ℹ️",
        )
        manual_key = st.sidebar.text_input(
            f"🔑 {provider} API Key 입력",
            type="password",
            placeholder="API Key...",
            help="세션에서만 사용되며 저장되지 않습니다.",
        )
        api_key = manual_key.strip()

    st.sidebar.divider()
    st.sidebar.caption("📋 현재 설정")
    if api_key:
        key_source = "🔐 .env" if has_env_key else "🔑 수동입력"
        key_status = f"✅ 사용 가능 ({key_source})"
    else:
        key_status = "❌ 미설정"

    st.sidebar.code(
        f"Provider: {provider}\nModel: {model_label}\nReasoning: {reasoning_level}\nAPI Key: {key_status}",
        language=None,
    )

    return LLMConfig(provider, api_key, model_id, reasoning_level) if api_key else None


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def clear_results():
    """데이터 변경 시 이전 분석 결과 초기화."""
    if "final_report" in st.session_state:
        del st.session_state["final_report"]
        st.toast("⚠️ 데이터 변경으로 이전 분석 결과가 초기화되었습니다.", icon="🔄")


def create_pdf(markdown_text: str) -> bytes:
    """Markdown 텍스트를 PDF로 변환 (Korean Font 포함)."""
    # 1. Markdown -> HTML
    html_content = markdown.markdown(markdown_text, extensions=['tables'])

    # 2. Add Korean Font Style (using Google Fonts Early Access for reliability)
    #    NanumGothic is commonly used.
    full_html = f"""
    <html>
    <head>
        <style>
            @font-face {{
                font-family: 'NanumGothic';
                src: url('https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf');
            }}
            body {{
                font-family: 'NanumGothic', sans-serif;
                font-size: 10pt;
                line-height: 1.6;
            }}
            h1, h2, h3 {{ color: #2E86C1; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """

    # 3. HTML -> PDF
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(full_html, dest=pdf_buffer)
    
    if pisa_status.err:
        return b""
    
    return pdf_buffer.getvalue()


# ─────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────

def section_upload():
    """섹션 1: 조서 PDF 업로드 및 파싱."""
    st.header("📄 1. 조서 업로드", divider="blue")

    uploaded_file = st.file_uploader(
        "수사 조서 PDF 파일을 업로드하세요",
        type=["pdf"],
        help="지원 형식: PDF (피의자 신문조서, 진술조서 등)",
        on_change=clear_results,  # 파일 변경 시 결과 초기화
    )

    if uploaded_file is not None:
        # 파일이 변경되었거나 아직 파싱되지 않았으면 파싱 실행
        if (
            "uploaded_filename" not in st.session_state
            or st.session_state.uploaded_filename != uploaded_file.name
        ):
            with st.status("📄 PDF 파싱 중...", expanded=True) as status:
                st.write("텍스트 추출 중...")
                raw_text = extract_text(uploaded_file)
                st.session_state.raw_text = raw_text

                st.write("문답(Q&A) 구조화 중...")
                parsed_df = parse_qa(raw_text)
                st.session_state.parsed_df = parsed_df
                st.session_state.uploaded_filename = uploaded_file.name
                
                # 파싱 완료 시에도 결과 초기화 확인
                clear_results()

                st.write(f"✅ 총 {len(parsed_df)}개 문답 추출 완료")
                status.update(label="파싱 완료!", state="complete", expanded=False)

        st.success(
            f"📂 **{uploaded_file.name}** — {len(st.session_state.parsed_df)}개 문답 추출됨"
        )

    return uploaded_file is not None and "parsed_df" in st.session_state


def section_review():
    """섹션 2: 데이터 확인 및 수정."""
    st.header("📝 2. 데이터 확인 및 수정", divider="orange")

    if "parsed_df" not in st.session_state:
        st.info("먼저 PDF 파일을 업로드하세요.")
        return False

    st.caption("💡 표에서 데이터 수정 시 분석 결과가 초기화됩니다.")

    edited_df = st.data_editor(
        st.session_state.parsed_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "index": st.column_config.NumberColumn("번호", width="small"),
            "type": st.column_config.SelectboxColumn("유형", options=["Q", "A"], width="small"),
            "speaker": st.column_config.SelectboxColumn("화자", options=["수사관", "피의자"], width="small"),
            "content": st.column_config.TextColumn("내용", width="large"),
        },
        key="data_editor",
        on_change=clear_results,  # 데이터 수정 시 결과 초기화
    )

    st.session_state.edited_df = edited_df

    # CSV 다운로드
    current_date = datetime.now(KST).strftime("%Y%m%d")
    file_name = f"범죄분석 선별 체크 결과_{current_date}.csv"
    csv_data = edited_df.to_csv(index=False, encoding="utf-8-sig")

    col1, col2 = st.columns([1, 5])
    with col1:
        st.download_button(
            label="💾 CSV 다운로드",
            data=csv_data,
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.caption(f"총 {len(edited_df)}개 행 | 📁 브라우저 다운로드 폴더 저장")

    return len(edited_df) > 0


def section_analysis(config: Optional[LLMConfig]):
    """섹션 3: AI 분석 실행 및 결과."""
    st.header("🔍 3. AI 분석 결과", divider="red")

    if config is None:
        st.warning("⚠️ 사이드바에서 API Key 설정 필요")
        return

    if "edited_df" not in st.session_state:
        st.info("먼저 데이터 확인 단계를 완료하세요.")
        return

    df = st.session_state.edited_df

    # 분석 시작 버튼
    if st.button("▶️ 분석 및 선별 시작", type="primary", use_container_width=True):
        _run_analysis(df, config)
        # _run_analysis 내부에서 st.rerun() 호출됨

    # 결과 표시 (Button 클릭과 무관하게 State 존재 시 표시)
    if "final_report" in st.session_state:
        st.divider()
        st.markdown(st.session_state.final_report)

        # PDF 다운로드
        current_date_str = datetime.now(KST).strftime("%Y%m%d")
        pdf_filename = f"범죄분석 선별 체크 결과_{current_date_str}.pdf"
        
        pdf_bytes = create_pdf(st.session_state.final_report)
        if pdf_bytes:
            st.download_button(
                label="📄 PDF 보고서 다운로드",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
            )
        else:
            st.error("PDF 생성 중 오류가 발생했습니다.")

        # 로그 표시
        if "analysis_log" in st.session_state:
            with st.expander("📋 분석 로그 상세 보기"):
                for log in st.session_state.analysis_log:
                    st.markdown(log)


def _run_analysis(df: pd.DataFrame, config: LLMConfig):
    """분석 실행 (Analyst → Critic → Reporter)."""
    chunks = create_chunks(df, size=20, overlap=3)
    total_chunks = len(chunks)

    if total_chunks == 0:
        st.error("분석할 데이터가 없습니다.")
        return

    all_verified = []
    analysis_log = []
    total_rejected = 0

    progress_bar = st.progress(0, text="분석 준비 중...")
    
    # 상태값 초기화
    if "final_report" in st.session_state:
        del st.session_state["final_report"]

    with st.status(f"🔄 총 {total_chunks}개 청크 분석 중...", expanded=True) as status:
        for i, chunk in enumerate(chunks):
            chunk_label = f"[청크 {i + 1}/{total_chunks}]"
            
            # Analyst
            st.write(f"{chunk_label} 🔍 분석 중...")
            try:
                draft = call_analyst(chunk, config)
                st.write(f"{chunk_label} → 추출 완료")
            except Exception as e:
                st.error(f"{chunk_label} 오류: {e}")
                continue

            # Critic
            st.write(f"{chunk_label} ✅ 검증 중...")
            try:
                verified = call_critic(chunk, draft, config)
                verified_count = len(verified.get("verified_findings", []))
                rejected_count = len(verified.get("rejected_findings", []))
                total_rejected += rejected_count
                
                st.write(f"{chunk_label} → ✅ {verified_count}건, ❌ {rejected_count}건")
                analysis_log.append(f"**{chunk_label}** ✅ {verified_count} / ❌ {rejected_count}")

                if verified.get("verified_findings"):
                    all_verified.extend(verified["verified_findings"])
            except Exception as e:
                st.error(f"{chunk_label} 검증 오류: {e}")
                continue

            progress_bar.progress((i + 1) / total_chunks, text=f"{chunk_label} 완료")

        status.update(label="✅ 분석 완료! 보고서 작성 중...", state="complete", expanded=False)

    # Reporter
    if all_verified:
        progress_bar.progress(1.0, text="📝 최종 보고서 작성 중...")
        try:
            verified_json = json.dumps(all_verified, ensure_ascii=False, indent=2)
            final_report = call_reporter(verified_json, config)
            
            # 결과 저장
            st.session_state.final_report = final_report
            st.session_state.analysis_log = analysis_log
            
            # 중요: 중복 출력을 막기 위해 여기서 출력하지 않고 Rerun
            st.rerun()
            
        except Exception as e:
            st.error(f"Reporter 오류: {e}")
    else:
        st.warning("발견된 특이사항이 없습니다.")
    
    progress_bar.empty()


def main():
    init_page()
    st.title("🔍 CASS Lite — 범죄분석 선별 시스템")
    st.caption("PDF 조서 → AI 분석 → 체크리스트 기반 보고서")
    st.divider()

    config = setup_sidebar()
    
    has_data = section_upload()
    if has_data:
        data_ready = section_review()
        if data_ready:
            section_analysis(config)


if __name__ == "__main__":
    main()
