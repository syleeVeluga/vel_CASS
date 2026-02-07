"""
청킹 모듈 단위 테스트 — 분할 크기, 오버랩, 엣지 케이스 검증.
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.chunker import create_chunks


def _make_df(n: int) -> pd.DataFrame:
    """테스트용 DataFrame 생성 (n개 행)."""
    return pd.DataFrame({
        "index": list(range(1, n + 1)),
        "type": ["Q" if i % 2 == 0 else "A" for i in range(n)],
        "speaker": ["수사관" if i % 2 == 0 else "피의자" for i in range(n)],
        "content": [f"내용 {i+1}" for i in range(n)],
    })


class TestCreateChunks:
    """create_chunks 함수 테스트."""

    def test_single_chunk(self):
        """20개 이하면 청크 1개."""
        df = _make_df(15)
        chunks = create_chunks(df, size=20, overlap=3)
        assert len(chunks) == 1
        assert "[Q1]" in chunks[0] or "[A1]" in chunks[0]

    def test_exact_size(self):
        """정확히 20개면 청크 1개."""
        df = _make_df(20)
        chunks = create_chunks(df, size=20, overlap=3)
        assert len(chunks) == 1

    def test_two_chunks_with_overlap(self):
        """25개면 청크 2개 (오버랩)."""
        df = _make_df(25)
        chunks = create_chunks(df, size=20, overlap=3)
        assert len(chunks) == 2
        # 두 번째 청크에 오버랩 데이터 포함 확인
        assert "내용 18" in chunks[1]  # 오버랩 영역

    def test_empty_dataframe(self):
        """빈 DataFrame."""
        df = pd.DataFrame(columns=["index", "type", "speaker", "content"])
        chunks = create_chunks(df)
        assert len(chunks) == 0

    def test_large_dataset(self):
        """100개 행 청킹 — 적절한 수의 청크 생성."""
        df = _make_df(100)
        chunks = create_chunks(df, size=20, overlap=3)
        # 100개, stride=17: 0,17,34,51,68,85 → 6개 청크
        assert len(chunks) >= 5
        assert len(chunks) <= 6

    def test_overlap_content_present(self):
        """오버랩 영역의 내용이 다음 청크에 존재하는지 확인."""
        df = _make_df(50)
        chunks = create_chunks(df, size=20, overlap=3)
        assert len(chunks) >= 2
        # 두 번째 청크에 오버랩 영역(18,19,20)이 포함되어야 함
        for content_num in [18, 19, 20]:
            assert f"내용 {content_num}" in chunks[1]

    def test_chunk_text_format(self):
        """청크 텍스트가 [Q/A##] 형식인지."""
        df = _make_df(5)
        chunks = create_chunks(df, size=20, overlap=3)
        assert len(chunks) == 1
        # 형식 확인
        for line in chunks[0].split("\n"):
            assert line.startswith("[Q") or line.startswith("[A")
