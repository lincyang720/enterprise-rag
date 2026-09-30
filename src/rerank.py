"""
重排序: 交叉编码器 (BAAI/bge-reranker-base)
对融合后的候选片段重新打分, 提升相关性
"""
import os

from .config import RERANK, EMB
from .utils import get_logger

if EMB.get("hf_mirror"):
    os.environ.setdefault("HF_ENDPOINT", EMB["hf_mirror"])

from sentence_transformers import CrossEncoder  # noqa: E402

LOG = get_logger()
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        LOG.info("加载重排序模型: %s", RERANK["model_name"])
        _MODEL = CrossEncoder(RERANK["model_name"], device=RERANK["device"])
    return _MODEL


def rerank(query: str, passages: list, top_n: int = None) -> list:
    """
    passages: [{"chunk_id","content","file_name","rel_dir","file_type","score",...}]
    返回按重排分数降序的前 top_n 个(附带 rerank_score)
    """
    if not passages:
        return []
    if top_n is None:
        top_n = RERANK["final_k"]
    model = get_model()
    pairs = [(query, p.get("content", "")) for p in passages]
    scores = model.predict(pairs, show_progress_bar=False)
    for p, s in zip(passages, scores):
        p["rerank_score"] = float(s)
    passages.sort(key=lambda x: x["rerank_score"], reverse=True)
    return passages[:top_n]
