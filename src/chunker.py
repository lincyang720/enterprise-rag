"""
文本分块: 滑动窗口 + 重叠
- 优先按段落切, 但统一用字符滑动窗口保证长度可控
- 返回 chunk 文本列表
"""
from .utils import clean_text


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100):
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        if piece.strip():
            chunks.append(piece.strip())
        if end >= len(text):
            break
        start += step
    return chunks
