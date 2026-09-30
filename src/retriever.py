"""
混合检索: BM25(关键词) + 向量余弦(语义) -> 加权融合 -> 重排序
- BM25: rank_bm25, 中文用 jieba 分词
- 向量: Chroma 余弦检索
- 融合: 各方法分数 min-max 归一化后加权求和
- 重排: cross-encoder
并封装为 LangChain BaseRetriever(通用 agent 框架兼容)
"""
import os
import pickle

from rank_bm25 import BM25Okapi

from .config import RET, RERANK
from .utils import tokenize, get_logger, stable_id
from . import store, embeddings, rerank

LOG = get_logger()

# ---------------- BM25 索引 ----------------
class BM25Index:
    def __init__(self):
        self.corpus = []        # list of token lists
        self.chunk_ids = []     # 与 corpus 对齐
        self.bm25 = None

    def build(self, chunks):
        """chunks: [(chunk_id, content)]"""
        self.chunk_ids = [c[0] for c in chunks]
        self.corpus = [tokenize(c[1]) for c in chunks]
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        LOG.info("BM25 索引构建完成, 共 %d 个片段", len(self.corpus))

    def search(self, query: str, topk: int = 30):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        # 取 topk
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [(self.chunk_ids[i], float(scores[i])) for i in idx if scores[i] > 0]


def _minmax(vals):
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


# ---------------- 混合检索 ----------------
class HybridSearch:
    def __init__(self):
        self.bm25_index = BM25Index()
        self._load_or_build_bm25()

    def _load_or_build_bm25(self):
        cache = store.DATA["bm25_cache"]
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as f:
                    count, self.bm25_index = pickle.load(f)
                if count == store.count_chunks():
                    LOG.info("从缓存加载 BM25 索引 (片段数 %d)", count)
                    return
                LOG.info("BM25 缓存与当前片段数不符, 重建")
            except Exception:
                pass
        chunks = store.get_all_chunks()
        self.bm25_index.build(chunks)
        try:
            with open(cache, "wb") as f:
                pickle.dump((len(chunks), self.bm25_index), f)
        except Exception:
            pass

    def reload_bm25(self):
        chunks = store.get_all_chunks()
        self.bm25_index.build(chunks)
        try:
            with open(store.DATA["bm25_cache"], "wb") as f:
                pickle.dump((len(chunks), self.bm25_index), f)
        except Exception:
            pass

    def _fuse(self, query: str, top_k: int = None, filter_rel_dir: str = None,
              filter_file_type: str = None):
        """BM25 + 向量余弦 -> 加权融合, 返回融合后的候选列表(截断到 top_k, 未重排)"""
        top_k = top_k or RERANK["rerank_candidates"]

        # 1) 向量检索
        qvec = embeddings.embed_query(query)
        where = {}
        if filter_rel_dir:
            where["rel_dir"] = filter_rel_dir
        if filter_file_type:
            where["file_type"] = filter_file_type
        col = store.get_chroma()
        try:
            vres = col.query(
                query_embeddings=[qvec],
                n_results=RET["vector_topk"],
                where=where if where else None,
                include=["distances", "metadatas", "documents"])
        except Exception as e:
            LOG.warning("向量检索失败: %s", e)
            vres = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        v_ids = vres["ids"][0]
        v_dist = vres["distances"][0]
        v_meta = vres["metadatas"][0]
        v_doc = vres["documents"][0]
        # 距离 -> 相似度(余弦距离=1-余弦)
        v_sim = [1 - d for d in v_dist]
        v_norm = _minmax(v_sim)
        v_map = {}
        for i, cid in enumerate(v_ids):
            v_map[cid] = {
                "chunk_id": cid,
                "content": v_doc[i] if i < len(v_doc) else "",
                "file_name": (v_meta[i] or {}).get("file_name", ""),
                "rel_dir": (v_meta[i] or {}).get("rel_dir", ""),
                "file_type": (v_meta[i] or {}).get("file_type", ""),
                "vector_score": v_sim[i],
                "vector_norm": v_norm[i] if i < len(v_norm) else 0.0,
            }

        # 2) BM25 检索
        bres = self.bm25_index.search(query, RET["bm25_topk"])
        b_norm = _minmax([s for _, s in bres])
        for i, (cid, score) in enumerate(bres):
            if cid in v_map:
                v_map[cid]["bm25_score"] = score
                v_map[cid]["bm25_norm"] = b_norm[i] if i < len(b_norm) else 0.0
            else:
                # BM25 命中的, 向量未命中, 需要补 content/元数据
                extra = self._content_by_id(cid)
                v_map[cid] = {
                    "chunk_id": cid,
                    "content": extra.get("content", ""),
                    "file_name": extra.get("file_name", ""),
                    "rel_dir": extra.get("rel_dir", ""),
                    "file_type": extra.get("file_type", ""),
                    "vector_score": 0.0,
                    "vector_norm": 0.0,
                    "bm25_score": score,
                    "bm25_norm": b_norm[i] if i < len(b_norm) else 0.0,
                }

        # 3) 加权融合
        wb = RET.get("bm25_weight", 0.5)
        wv = RET.get("vector_weight", 0.5)
        candidates = []
        for cid, m in v_map.items():
            bn = m.get("bm25_norm", 0.0)
            vn = m.get("vector_norm", 0.0)
            m["fusion_score"] = wb * bn + wv * vn
            candidates.append(m)
        candidates.sort(key=lambda x: x["fusion_score"], reverse=True)
        return candidates[:top_k]

    def retrieve_candidates(self, query: str, top_k: int = None,
                            filter_rel_dir: str = None, filter_file_type: str = None):
        """仅做融合检索(不重排), 供 LangGraph 图节点调用"""
        return self._fuse(query, top_k, filter_rel_dir, filter_file_type)

    def search(self, query: str, top_k: int = None, filter_rel_dir: str = None,
               filter_file_type: str = None):
        """端到端: 融合 -> 重排序, 返回最终 top-k(带 rerank_score)"""
        candidates = self._fuse(query, top_k, filter_rel_dir, filter_file_type)
        reranked = rerank.rerank(query, candidates, RERANK["final_k"])
        return reranked

    def _content_by_id(self, chunk_id):
        # 从 chroma 取 (BM25 独有候选), 同时补全元数据
        try:
            col = store.get_chroma()
            r = col.get(ids=[chunk_id], include=["documents", "metadatas"])
            if r and r["documents"]:
                meta = (r["metadatas"][0] or {}) if r.get("metadatas") else {}
                return {
                    "content": r["documents"][0],
                    "file_name": meta.get("file_name", ""),
                    "rel_dir": meta.get("rel_dir", ""),
                    "file_type": meta.get("file_type", ""),
                }
        except Exception:
            pass
        return {"content": "", "file_name": "", "rel_dir": "", "file_type": ""}


# ---------------- LangChain 兼容封装 ----------------
def build_langchain_retriever(hybrid: HybridSearch):
    try:
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from pydantic import ConfigDict

        class HybridRetriever(BaseRetriever):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            hybrid: HybridSearch

            def _get_relevant_documents(self, query, run_manager=None):
                results = self.hybrid.search(query)
                docs = []
                for r in results:
                    docs.append(Document(
                        page_content=r.get("content", ""),
                        metadata={
                            "source": r.get("file_name", ""),
                            "rel_dir": r.get("rel_dir", ""),
                            "file_type": r.get("file_type", ""),
                            "score": r.get("rerank_score", r.get("fusion_score", 0)),
                        }))
                return docs

        LOG.info("已构建 LangChain 兼容 HybridRetriever")
        return HybridRetriever(hybrid=hybrid)
    except Exception as e:
        LOG.warning("LangChain 封装不可用(不影响检索): %s", e)
        return None
