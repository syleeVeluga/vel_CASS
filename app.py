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

from parsing.pdf_parser import extract_text, parse_qa, save_csv
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


def init_page():
    """페이지 기본 설정."""
    st.set_page_config(
        page_title="CASS Lite — 범죄분석 선별 시스템",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def setup_sidebar() -> LLMConfig | None:
    """
    사이드바: 프로바이더, 모델, Reasoning 레벨, API Key 설정.
    Returns LLMConfig or None if not configured.
    """
    st.sidebar.title("⚙️ AI 설정")
    st.sidebar.divider()

    # 프로바이더 선택
    provider = st.sidebar.selectbox(
        "LLM 프로바이더",
        options=list(AVAILABLE_MODELS.keys()),
        help="분석에 사용할 AI 모델 프로바이더를 선택하세요.",
    )

    # 모델 선택
    model_options = AVAILABLE_MODELS[provider]
    model_label = st.sidebar.selectbox(
        "모델",
        options=list(model_options.keys()),
    )
    model_id = model_options[model_label]

    # Reasoning 레벨 선택
    if provider == "OpenAI":
        reasoning_options = REASONING_LEVELS["OpenAI"]
    else:
        reasoning_options = REASONING_LEVELS["Gemini"].get(model_id, ["low", "high"])

    reasoning_level = st.sidebar.select_slider(
        "Reasoning 레벨",
        options=reasoning_options,
        value=reasoning_options[len(reasoning_options) // 2],  # 중간값 기본
        help="높을수록 깊이 사고하지만 응답이 느려질 수 있습니다.",
    )

    st.sidebar.divider()

    # ── API Key 보안 처리 ──
    # 매번 .env 파일을 다시 읽어 최신 키를 반영 (서버 재시작 불필요)
    load_dotenv(_ENV_PATH, override=True)
    env_key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GOOGLE_API_KEY"

    # .env 파일에서 직접 읽기 (os.getenv가 빈 문자열을 반환하는 경우 대비)
    env_values = dotenv_values(_ENV_PATH)
    env_key = (env_values.get(env_key_name) or "").strip()
    if not env_key:
        # 시스템 환경변수 fallback
        env_key = (os.getenv(env_key_name) or "").strip()
    has_env_key = bool(env_key)

    api_key = ""

    if has_env_key:
        # .env에 키가 있으면 존재 여부만 표시 (값은 절대 노출 안 함)
        st.sidebar.success(
            f"🔐 **{env_key_name}** — .env에서 로드됨\n\n"
            f"키가 안전하게 설정되어 있습니다.",
            icon="✅",
        )
        api_key = env_key  # 내부적으로만 사용
    else:
        # .env에 키가 없으면 수동 입력 허용 (password 타입)
        st.sidebar.info(
            f"📝 .env에 `{env_key_name}`이 설정되지 않았습니다.\n\n"
            f"아래에서 직접 입력하세요.",
            icon="ℹ️",
        )
        manual_key = st.sidebar.text_input(
            f"🔑 {provider} API Key 입력",
            value="",
            type="password",
            placeholder="API Key를 입력하세요...",
            help="입력된 키는 현재 세션에서만 사용되며 저장되지 않습니다.",
        )
        api_key = manual_key.strip()

    # 설정 요약
    st.sidebar.divider()
    st.sidebar.caption("📋 현재 설정")

    if api_key:
        key_source = "🔐 .env" if has_env_key else "🔑 수동입력"
        key_status = f"✅ 사용 가능 ({key_source})"
    else:
        key_status = "❌ 미설정"

    st.sidebar.code(
        f"Provider: {provider}\n"
        f"Model: {model_label}\n"
        f"Reasoning: {reasoning_level}\n"
        f"API Key: {key_status}",
        language=None,
    )

    if not api_key:
        return None

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model_id,
        reasoning_level=reasoning_level,
    )


def section_upload():
    """섹션 1: 조서 PDF 업로드 및 파싱."""
    st.header("📄 1. 조서 업로드", divider="blue")

    uploaded_file = st.file_uploader(
        "수사 조서 PDF 파일을 업로드하세요",
        type=["pdf"],
        help="지원 형식: PDF (피의자 신문조서, 진술조서 등)",
    )

    if uploaded_file is not None:
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

                st.write(f"✅ 총 {len(parsed_df)}개 문답 추출 완료")
                status.update(label="파싱 완료!", state="complete", expanded=False)

        st.success(
            f"📂 **{uploaded_file.name}** — "
            f"{len(st.session_state.parsed_df)}개 문답 추출됨"
        )

    return uploaded_file is not None and "parsed_df" in st.session_state


def section_review():
    """섹션 2: 파싱 데이터 확인 및 수정."""
    st.header("📝 2. 데이터 확인 및 수정", divider="orange")

    if "parsed_df" not in st.session_state:
        st.info("먼저 PDF 파일을 업로드하세요.")
        return False

    st.caption(
        "💡 아래 표에서 직접 수정할 수 있습니다. "
        "정규식이 놓친 문답 분리를 수정하면 분석 정확도가 향상됩니다."
    )

    edited_df = st.data_editor(
        st.session_state.parsed_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "index": st.column_config.NumberColumn("번호", width="small"),
            "type": st.column_config.SelectboxColumn(
                "유형", options=["Q", "A"], width="small"
            ),
            "speaker": st.column_config.SelectboxColumn(
                "화자", options=["수사관", "피의자"], width="small"
            ),
            "content": st.column_config.TextColumn("내용", width="large"),
        },
        key="data_editor",
    )

    # 수정된 데이터를 세션에 반영
    st.session_state.edited_df = edited_df

    # CSV 저장 버튼 (다운로드 폴더로 저장)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("💾 CSV 저장", use_container_width=True):
            saved_path = save_csv(edited_df)
            st.toast(f"✅ 다운로드 폴더에 저장 완료!\n{saved_path}", icon="💾")

    with col2:
        st.caption(f"총 {len(edited_df)}개 행 | 📁 다운로드 폴더에 저장됩니다")

    return len(edited_df) > 0


def section_analysis(config: LLMConfig | None):
    """섹션 3: AI 분석 실행 및 결과."""
    st.header("🔍 3. AI 분석 결과", divider="red")

    if config is None:
        st.warning("⚠️ 사이드바에서 API Key를 설정해주세요.")
        return

    if "edited_df" not in st.session_state:
        st.info("먼저 데이터 확인 단계를 완료하세요.")
        return

    df = st.session_state.edited_df

    # 분석 시작 버튼
    if st.button("▶️ 분석 및 선별 시작", type="primary", use_container_width=True):
        _run_analysis(df, config)

    # 이전 분석 결과가 있으면 표시
    if "final_report" in st.session_state:
        st.divider()
        st.markdown(st.session_state.final_report)

        # 분석 로그
        if "analysis_log" in st.session_state:
            with st.expander("📋 분석 로그 상세 보기"):
                for log_entry in st.session_state.analysis_log:
                    st.markdown(log_entry)


def _run_analysis(df: pd.DataFrame, config: LLMConfig):
    """분석 파이프라인 실행 (Analyst → Critic → Reporter)."""
    chunks = create_chunks(df, size=20, overlap=3)
    total_chunks = len(chunks)

    if total_chunks == 0:
        st.error("분석할 문답 데이터가 없습니다.")
        return

    all_verified = []
    analysis_log = []
    total_rejected = 0

    progress_bar = st.progress(0, text="분석 준비 중...")

    with st.status(f"🔄 총 {total_chunks}개 청크 분석 중...", expanded=True) as status:
        for i, chunk in enumerate(chunks):
            chunk_label = f"[청크 {i + 1}/{total_chunks}]"

            # ── Step A: Analyst ──
            st.write(f"{chunk_label} 🔍 분석 중...")
            try:
                draft = call_analyst(chunk, config)
                finding_count = sum(
                    len(draft.get(k, []))
                    for k in ["admissions", "contradictions", "alibis", "suspicious_indicators"]
                )
                st.write(f"{chunk_label} → {finding_count}건 발견")
                analysis_log.append(
                    f"**{chunk_label}** Analyst: {finding_count}건 추출"
                )
            except Exception as e:
                st.error(f"{chunk_label} Analyst 오류: {e}")
                analysis_log.append(f"**{chunk_label}** ❌ Analyst 오류: {e}")
                continue

            # ── Step B: Critic ──
            st.write(f"{chunk_label} ✅ 검증 중...")
            try:
                verified = call_critic(chunk, draft, config)
                verified_count = len(verified.get("verified_findings", []))
                rejected_count = len(verified.get("rejected_findings", []))
                total_rejected += rejected_count

                st.write(
                    f"{chunk_label} → {verified_count}건 통과, "
                    f"{rejected_count}건 기각"
                )
                analysis_log.append(
                    f"**{chunk_label}** Critic: ✅ {verified_count}건 통과 / "
                    f"❌ {rejected_count}건 기각"
                )

                if verified.get("verified_findings"):
                    all_verified.extend(verified["verified_findings"])
            except Exception as e:
                st.error(f"{chunk_label} Critic 오류: {e}")
                analysis_log.append(f"**{chunk_label}** ❌ Critic 오류: {e}")
                continue

            # 진행률 업데이트
            progress_bar.progress(
                (i + 1) / total_chunks,
                text=f"{chunk_label} 완료 ({i + 1}/{total_chunks})",
            )

        status.update(
            label=f"✅ 분석 완료 — {len(all_verified)}건 검증 통과, {total_rejected}건 기각",
            state="complete",
            expanded=False,
        )

    # ── Step C: Reporter ──
    if all_verified:
        progress_bar.progress(1.0, text="📝 최종 보고서 작성 중...")

        with st.status("📝 최종 보고서 작성 중...", expanded=True) as status:
            try:
                verified_json = json.dumps(all_verified, ensure_ascii=False, indent=2)
                final_report = call_reporter(verified_json, config)
                st.session_state.final_report = final_report
                st.session_state.analysis_log = analysis_log
                status.update(label="✅ 보고서 작성 완료", state="complete")
            except Exception as e:
                st.error(f"Reporter 오류: {e}")
                status.update(label="❌ 보고서 작성 실패", state="error")
                return

        # 결과 표시
        st.divider()
        st.markdown(final_report)
    else:
        st.warning("검증을 통과한 발견 사항이 없습니다.")

    progress_bar.empty()


def main():
    """메인 앱 실행."""
    init_page()

    st.title("🔍 CASS Lite — 범죄분석 선별 시스템")
    st.caption("PDF 조서 → AI 분석 → 체크리스트 기반 보고서")
    st.divider()

    # 사이드바 설정
    config = setup_sidebar()

    # 메인 3단계 워크플로우
    has_data = section_upload()
    if has_data:
        data_ready = section_review()
        if data_ready:
            section_analysis(config)


if __name__ == "__main__":
    main()
