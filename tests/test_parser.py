"""
PDF 파서 단위 테스트 — 정규식 매칭, 연속 병합, DataFrame 구조 검증.
"""

import pandas as pd
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parsing.pdf_parser import parse_qa, save_csv


class TestParseQA:
    """parse_qa 함수 테스트."""

    def test_basic_qa_parsing(self):
        """기본 문/답 패턴 분리."""
        text = "문: 이름이 무엇입니까?\n답: 홍길동입니다."
        df = parse_qa(text)
        assert len(df) == 2
        assert df.iloc[0]["type"] == "Q"
        assert df.iloc[1]["type"] == "A"
        assert df.iloc[0]["speaker"] == "수사관"
        assert df.iloc[1]["speaker"] == "피의자"

    def test_qa_without_colon(self):
        """콜론 없는 문/답 패턴."""
        text = "문 이름이 무엇입니까?\n답 홍길동입니다."
        df = parse_qa(text)
        assert len(df) == 2
        assert "이름이" in df.iloc[0]["content"]

    def test_continuation_merge(self):
        """패턴 미매칭 줄이 이전 발언에 병합되는지 확인."""
        text = "문: 당시 상황을 설명해주세요.\n답: 그날 저는 집에 있었습니다.\n아무것도 하지 않았습니다.\n그냥 TV를 봤습니다."
        df = parse_qa(text)
        assert len(df) == 2
        # 답변에 연속 줄이 병합되어야 함
        answer = df.iloc[1]["content"]
        assert "아무것도" in answer
        assert "TV를" in answer

    def test_empty_text(self):
        """빈 텍스트 처리."""
        df = parse_qa("")
        assert len(df) == 0
        assert list(df.columns) == ["index", "type", "speaker", "content"]

    def test_no_qa_pattern(self):
        """문/답 패턴이 없는 텍스트."""
        text = "이것은 일반 텍스트입니다.\n특별한 패턴이 없습니다."
        df = parse_qa(text)
        assert len(df) == 0

    def test_multiple_qa_pairs(self):
        """여러 문답 쌍 파싱."""
        text = "\n".join([
            "문: 질문1?",
            "답: 답변1.",
            "문: 질문2?",
            "답: 답변2.",
            "문: 질문3?",
            "답: 답변3.",
        ])
        df = parse_qa(text)
        assert len(df) == 6
        assert df.iloc[0]["index"] == 1
        assert df.iloc[5]["index"] == 6

    def test_dataframe_columns(self):
        """DataFrame 컬럼 구조 검증."""
        text = "문: 테스트\n답: 응답"
        df = parse_qa(text)
        expected_cols = ["index", "type", "speaker", "content"]
        assert list(df.columns) == expected_cols


class TestSaveCSV:
    """save_csv 함수 테스트."""

    def test_save_creates_file(self, tmp_path):
        """CSV 파일 생성 확인."""
        df = pd.DataFrame({
            "index": [1, 2],
            "type": ["Q", "A"],
            "speaker": ["수사관", "피의자"],
            "content": ["질문", "답변"],
        })
        output_path = str(tmp_path / "output" / "test.csv")
        saved = save_csv(df, output_path)
        assert os.path.exists(saved)

    def test_save_content(self, tmp_path):
        """저장된 CSV 내용 검증."""
        df = pd.DataFrame({
            "index": [1],
            "type": ["Q"],
            "speaker": ["수사관"],
            "content": ["테스트 질문입니다"],
        })
        output_path = str(tmp_path / "test.csv")
        save_csv(df, output_path)
        loaded = pd.read_csv(output_path)
        assert loaded.iloc[0]["content"] == "테스트 질문입니다"
