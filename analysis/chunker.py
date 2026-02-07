"""
스마트 청킹 모듈 — 문답 DataFrame을 분석용 청크로 분할합니다.
20개 Q&A 단위 + 3개 오버랩으로 문맥 단절을 방지합니다.
"""

import pandas as pd
from typing import List


def create_chunks(df: pd.DataFrame, size: int = 20, overlap: int = 3) -> List[str]:
    """
    DataFrame을 텍스트 청크 리스트로 변환.
    
    Args:
        df: parse_qa()의 결과 DataFrame (columns: index, type, speaker, content)
        size: 청크당 Q&A 개수 (기본: 20)
        overlap: 오버랩 Q&A 개수 (기본: 3)
    
    Returns:
        List[str] — 각 청크의 텍스트 (문답 형식)
    """
    if df.empty:
        return []
    
    total_rows = len(df)
    chunks = []
    start = 0
    stride = size - overlap
    
    # stride가 0 이하면 오버랩이 과도한 것이므로 보정
    if stride <= 0:
        stride = max(1, size)
    
    while start < total_rows:
        end = min(start + size, total_rows)
        chunk_df = df.iloc[start:end]
        
        # 청크를 텍스트로 변환
        chunk_text = _dataframe_to_text(chunk_df)
        chunks.append(chunk_text)
        
        # 이미 전체를 포함했으면 종료
        if end >= total_rows:
            break
        
        # 다음 청크 시작점 (오버랩 적용)
        next_start = start + stride
        
        # 남은 행이 오버랩 이하이면 현재 청크에서 끝냄
        remaining = total_rows - next_start
        if remaining <= overlap:
            break
        
        start = next_start
    
    return chunks


def _dataframe_to_text(df: pd.DataFrame) -> str:
    """
    DataFrame 슬라이스를 문답 텍스트로 변환.
    
    출력 형식:
        [Q1] 수사관: 질문 내용...
        [A1] 피의자: 답변 내용...
    """
    lines = []
    for _, row in df.iterrows():
        idx = row["index"]
        qa_type = row["type"]
        speaker = row["speaker"]
        content = row["content"]
        lines.append(f"[{qa_type}{idx}] {speaker}: {content}")
    
    return "\n".join(lines)
