"""
PDF 파싱 모듈 — docling을 사용하여 PDF에서 텍스트를 추출하고,
정규식으로 문답(Q&A)을 구조화합니다.
"""

import re
import os
import tempfile
import pandas as pd
from pathlib import Path
from typing import Optional


def extract_text(uploaded_file) -> str:
    """
    Streamlit UploadedFile 또는 파일 경로에서 docling을 사용하여 텍스트 추출.
    
    Args:
        uploaded_file: Streamlit UploadedFile 객체 또는 파일 경로 문자열
    
    Returns:
        추출된 텍스트 문자열
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    # OCR 및 테이블 구조 분석 비활성화 (텍스트 기반 조서에 불필요, Cloud 호환성)
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
    )

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    # Streamlit UploadedFile인 경우 임시 파일로 저장
    if hasattr(uploaded_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            result = converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            os.unlink(tmp_path)
    else:
        # 파일 경로 문자열인 경우
        result = converter.convert(str(uploaded_file))
        return result.document.export_to_markdown()


def parse_qa(raw_text: str) -> pd.DataFrame:
    """
    원시 텍스트에서 문답(Q&A)을 구조화된 DataFrame으로 변환.
    
    정규식 r"^(문|답)\s*[:]?\s*(.*)" 을 적용하여:
    - '문' → type='Q' (질문)
    - '답' → type='A' (답변)
    - 패턴 미매칭 → 이전 발언의 연속으로 병합
    
    Returns:
        DataFrame with columns: [index, type, speaker, content]
    """
    pattern = re.compile(r"^(문|답)\s*[:]?\s*(.*)", re.MULTILINE)
    
    records = []
    current_record = None
    qa_index = 0
    
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        match = pattern.match(line)
        if match:
            # 이전 레코드가 있으면 저장
            if current_record is not None:
                records.append(current_record)
            
            qa_type = "Q" if match.group(1) == "문" else "A"
            speaker = "수사관" if qa_type == "Q" else "피의자"
            qa_index += 1
            
            current_record = {
                "index": qa_index,
                "type": qa_type,
                "speaker": speaker,
                "content": match.group(2).strip(),
            }
        else:
            # 패턴 미매칭 → 이전 발언에 연속 병합
            if current_record is not None:
                current_record["content"] += " " + line
    
    # 마지막 레코드 저장
    if current_record is not None:
        records.append(current_record)
    
    if not records:
        return pd.DataFrame(columns=["index", "type", "speaker", "content"])
    
    return pd.DataFrame(records)


def _get_downloads_folder() -> Path:
    """
    OS에 관계없이 사용자의 다운로드 폴더 경로를 반환.
    Windows: C:/Users/<user>/Downloads
    macOS/Linux: /Users/<user>/Downloads
    """
    return Path.home() / "Downloads"


def save_csv(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """
    DataFrame을 CSV 파일로 저장.
    
    Args:
        df: 저장할 DataFrame
        path: 저장 경로 (기본: ~/Downloads/parsed_current.csv)
    
    Returns:
        저장된 파일의 절대 경로
    """
    if path is None:
        path = str(_get_downloads_folder() / "parsed_current.csv")

    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(Path(path).resolve())
