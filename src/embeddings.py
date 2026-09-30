"""
本地向量化 (sentence-transformers)
- 文档/查询分别编码; bge 系列查询需加检索指令前缀
- 走 HF 镜像加速国内下载
"""
import os

from .config import EMB

if EMB.get("hf_mirror"):
    os.environ.setdefault("HF_ENDPOINT", EMB["hf_mirror"])

from sentence_transformers import SentenceTransformer  # noqa: E402

_MODEL = None
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def get_model():
    global _MODEL
    if _MODEL is None:
        LOG.info("加载 embedding 模型: %s", EMB["model_name"])
        _MODEL = SentenceTransformer(EMB["model_name"], device=EMB["device"])
    return _MODEL


def embed_documents(texts: list) -> list:
    if not texts:
        return []
    model = get_model()
    B = 256
    out = []
    for i in range(0, len(texts), B):
        vecs = model.encode(
            texts[i:i + B],
            normalize_embeddings=EMB["normalize_embeddings"],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        out.extend(vecs.tolist())
    return out


def embed_query(text: str) -> list:
    model = get_model()
    vec = model.encode(
        _QUERY_PREFIX + text,
        normalize_embeddings=EMB["normalize_embeddings"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vec.tolist()


# 延迟导入日志, 避免循环
from .utils import get_logger  # noqa: E402
LOG = get_logger()
