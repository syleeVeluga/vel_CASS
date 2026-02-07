"""
LLM 유틸리티 모듈 — OpenAI GPT-5.2 및 Google Gemini 3 지원.
Analyst → Critic → Reporter 파이프라인 함수 제공.
"""

import json
from typing import Optional
from dataclasses import dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .prompts import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_TEMPLATE,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_TEMPLATE,
    REPORTER_SYSTEM_PROMPT,
    REPORTER_USER_TEMPLATE,
)


# ─────────────────────────────────────────────
# 지원 모델 정의
# ─────────────────────────────────────────────
AVAILABLE_MODELS = {
    "OpenAI": {
        "GPT-5.2": "gpt-5.2",
    },
    "Gemini": {
        "Gemini 3 Flash": "gemini-3-flash-preview",
        "Gemini 3 Pro": "gemini-3-pro-preview",
    },
}

# Reasoning/Thinking 레벨 정의
REASONING_LEVELS = {
    "OpenAI": ["low", "medium", "high"],
    "Gemini": {
        "gemini-3-flash-preview": ["minimal", "low", "medium", "high"],
        "gemini-3-pro-preview": ["low", "high"],
    },
}


@dataclass
class LLMConfig:
    """LLM 연결 설정"""
    provider: str       # "OpenAI" or "Gemini"
    api_key: str
    model: str          # 모델 API 이름
    reasoning_level: str = "medium"  # reasoning/thinking 레벨


# Retry 설정: 4초 ~ 60초 사이 지수 백오프, 최대 5회 재시도
_RETRY_CONFIG = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential(multiplier=1, min=4, max=60),
    "retry": retry_if_exception_type(Exception),
}


@retry(**_RETRY_CONFIG)
def _call_openai(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """OpenAI GPT-5.2 API 호출."""
    from openai import OpenAI

    client = OpenAI(api_key=config.api_key)

    # reasoning_effort 파라미터 설정
    kwargs = {
        "model": config.model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    # GPT-5.2 시리즈는 reasoning_effort 지원
    if config.reasoning_level and config.reasoning_level in ["low", "medium", "high"]:
        kwargs["reasoning"] = {"effort": config.reasoning_level}

    response = client.responses.create(**kwargs)
    return response.output_text


@retry(**_RETRY_CONFIG)
def _call_gemini(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """Google Gemini 3 API 호출."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.api_key)

    # thinking_level 설정
    thinking_config = None
    if config.reasoning_level:
        thinking_config = types.ThinkingConfig(
            thinking_level=config.reasoning_level.upper()
        )

    response = client.models.generate_content(
        model=config.model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=thinking_config,
        ),
    )
    return response.text


def _call_llm(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """프로바이더에 따라 적절한 LLM API 호출."""
    if config.provider == "OpenAI":
        return _call_openai(config, system_prompt, user_prompt)
    elif config.provider == "Gemini":
        return _call_gemini(config, system_prompt, user_prompt)
    else:
        raise ValueError(f"지원하지 않는 프로바이더: {config.provider}")


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON 블록 추출 및 파싱."""
    # ```json ... ``` 블록 추출 시도
    import re
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 빈 결과 반환
        return {
            "admissions": [],
            "contradictions": [],
            "alibis": [],
            "suspicious_indicators": [],
        }


# ─────────────────────────────────────────────
# 파이프라인 함수
# ─────────────────────────────────────────────

def call_analyst(chunk_text: str, config: LLMConfig) -> dict:
    """
    Analyst 단계: 청크에서 혐의점, 모순, 알리바이, 의심 지표 추출.
    
    Args:
        chunk_text: 분석할 문답 청크 텍스트
        config: LLM 설정
    
    Returns:
        dict — 분석 결과 JSON
    """
    user_prompt = ANALYST_USER_TEMPLATE.format(chunk_text=chunk_text)
    response = _call_llm(config, ANALYST_SYSTEM_PROMPT, user_prompt)
    return _extract_json(response)


def call_critic(source_text: str, draft_json: dict, config: LLMConfig) -> dict:
    """
    Critic 단계: Analyst 결과를 원본 텍스트와 대조하여 검증.
    
    Args:
        source_text: 원본 청크 텍스트
        draft_json: Analyst가 추출한 결과 JSON
        config: LLM 설정
    
    Returns:
        dict — 검증 결과 (verified_findings + rejected_findings)
    """
    user_prompt = CRITIC_USER_TEMPLATE.format(
        draft_json=json.dumps(draft_json, ensure_ascii=False, indent=2),
        source_text=source_text,
    )
    response = _call_llm(config, CRITIC_SYSTEM_PROMPT, user_prompt)
    result = _extract_json(response)

    # 기본 구조 보장
    if "verified_findings" not in result:
        result["verified_findings"] = []
    if "rejected_findings" not in result:
        result["rejected_findings"] = []

    return result


def call_reporter(verified_facts: str, config: LLMConfig) -> str:
    """
    Reporter 단계: 검증된 사실을 체크리스트 포함 최종 Markdown 리포트로 종합.
    
    Args:
        verified_facts: 전체 검증 완료 결과 (JSON 텍스트)
        config: LLM 설정
    
    Returns:
        str — 한국어 Markdown 리포트
    """
    user_prompt = REPORTER_USER_TEMPLATE.format(verified_facts=verified_facts)
    return _call_llm(config, REPORTER_SYSTEM_PROMPT, user_prompt)
