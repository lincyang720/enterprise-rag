"""
通用工具: 日志、文件指纹、中文分词、文本清洗
"""
import os
import hashlib
import logging
import re

import jieba

jieba.setLogLevel(logging.ERROR)

# ---------- 日志 ----------
def get_logger(name="bjhc_rag"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

LOG = get_logger()

# ---------- 文件指纹 (用于断点续传) ----------
def file_fingerprint(path: str):
    """返回 (mtime, size) 用于判断文件是否变化"""
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except Exception:
        return (0.0, 0)

def stable_id(*parts):
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h

# ---------- 中文分词 (BM25 用) ----------
def tokenize(text: str):
    text = clean_text(text)
    # jieba 对中文切词, 英文/数字保留原词
    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    # 过滤纯标点/空白
    tokens = [t for t in tokens if re.search(r"[\w\u4e00-\u9fff]", t)]
    return tokens

# ---------- 文本清洗 ----------
def clean_text(text: str) -> str:
    if not text:
        return ""
    # 去除多余空白
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_printable(text: str, min_len: int = 2) -> str:
    """从乱码/二进制残片中只保留可打印中英文片段"""
    # 保留 CJK、字母、数字、常见标点、换行
    keep = re.findall(
        r"[\u4e00-\u9fffA-Za-z0-9%+\-./:：，。、；;（）()【】\[\]“”\"'·\s]{1,}",
        text)
    chunks = [c.strip() for c in keep if len(c.strip()) >= min_len]
    return "\n".join(chunks)
