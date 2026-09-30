"""
LangGraph 编排的 RAG 问答流水线
================================
图结构(StateGraph):
    retrieve  --融合检索(BM25+向量)-->  rerank  --交叉编码器重排-->
    grade     --相关性筛选(条件边)-->   generate (有相关文档)
                                     -> no_answer (无相关文档)

编译后的 graph 本身是一个 LangChain Runnable, 可直接:
    - graph.invoke({"question": "..."})
    - 接入 LangChain / LangGraph 的 Agent / 工具调用
    - 用 .stream() 观察每个节点的中间结果(便于调试与评测)

入口便捷函数: run_rag(question, ...) -> {answer, sources, note, trace}
"""
from typing import TypedDict, Optional, List, Dict, Any
import time

from langgraph.graph import StateGraph, END

from .config import LLM, RERANK, RET, OBS
from . import rerank as rerank_mod
from . import llm as llm_mod
from . import store as store_mod
from .llm import is_llm_ready
from . import observability as obs
from .retriever import HybridSearch
from .utils import get_logger

LOG = get_logger()

# 全局懒加载检索器(图节点共享)
_HYBRID: Optional[HybridSearch] = None


def _get_hybrid() -> HybridSearch:
    global _HYBRID
    if _HYBRID is None:
        _HYBRID = HybridSearch()
    return _HYBRID


# ---------------- 状态 ----------------
class RAGState(TypedDict, total=False):
    question: str
    filter_rel_dir: Optional[str]
    filter_file_type: Optional[str]
    use_llm: bool
    candidates: List[Dict]      # 融合后候选(重排前)
    docs: List[Dict]            # 重排后 top-k
    graded: List[Dict]          # 相关性筛选后
    answer: Optional[str]
    note: str                   # 状态说明(如"未找到"/"未启用大模型")
    trace: Dict                 # 各节点统计, 供 UI/评测展示


# ---------------- 节点 ----------------
def retrieve_node(state: RAGState) -> Dict:
    t0 = time.perf_counter()
    hybrid = _get_hybrid()
    cands = hybrid.retrieve_candidates(
        state["question"],
        filter_rel_dir=state.get("filter_rel_dir"),
        filter_file_type=state.get("filter_file_type"),
    )
    trace = state.get("trace", {})
    trace["retrieved"] = len(cands)
    trace["t_retrieve"] = round(time.perf_counter() - t0, 3)
    return {"candidates": cands, "trace": trace}


def rerank_node(state: RAGState) -> Dict:
    t0 = time.perf_counter()
    docs = rerank_mod.rerank(
        state["question"], state.get("candidates", []), RERANK["final_k"])
    trace = state.get("trace", {})
    trace["reranked"] = len(docs)
    trace["top_rerank"] = round(max((d.get("rerank_score", 0) for d in docs), default=0), 3)
    trace["t_rerank"] = round(time.perf_counter() - t0, 3)
    return {"docs": docs, "trace": trace}


def grade_node(state: RAGState) -> Dict:
    t0 = time.perf_counter()
    docs = state.get("docs", [])
    threshold = RERANK.get("grade_threshold", 0.05)
    graded = [d for d in docs if d.get("rerank_score", 0) >= threshold]
    # 阈值过严导致全空时, 退而保留重排分最高的 1 条, 避免误报"未找到"
    if not graded and docs:
        graded = [max(docs, key=lambda d: d.get("rerank_score", 0))]
    trace = state.get("trace", {})
    trace["graded"] = len(graded)
    trace["t_grade"] = round(time.perf_counter() - t0, 3)
    return {"graded": graded, "trace": trace}


def _should_generate(state: RAGState) -> str:
    return "generate" if state.get("graded") else "no_answer"


def generate_node(state: RAGState) -> Dict:
    t0 = time.perf_counter()
    if not state.get("use_llm"):
        return {"answer": None, "note": "未启用大模型, 仅展示检索片段"}
    ans = llm_mod.answer_with_llm(state["question"], state.get("graded", []))
    trace = state.get("trace", {})
    trace["t_generate"] = round(time.perf_counter() - t0, 3)
    if ans:
        return {"answer": ans, "trace": trace}
    return {"answer": None,
            "note": "大模型未返回答案(可能未配置 Key), 仅展示检索片段",
            "trace": trace}


def no_answer_node(state: RAGState) -> Dict:
    return {"answer": None, "note": "资料中未找到相关信息", "graded": []}


# ---------------- 编译图 ----------------
def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("grade", grade_node)
    g.add_node("generate", generate_node)
    g.add_node("no_answer", no_answer_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges(
        "grade", _should_generate,
        {"generate": "generate", "no_answer": "no_answer"})
    g.add_edge("generate", END)
    g.add_edge("no_answer", END)
    return g.compile()


GRAPH = _build_graph()


def rebuild_bm25():
    """重建 BM25 缓存(索引扩充后调用, 让关键词检索覆盖新片段)。"""
    h = _get_hybrid()
    h.reload_bm25()
    return store_mod.count_chunks()


def run_rag(question: str, filter_rel_dir: str = None,
            filter_file_type: str = None, use_llm: bool = None) -> Dict:
    """便捷入口: 跑一遍图, 返回 {answer, sources, note, trace, question}"""
    if use_llm is None:
        use_llm = is_llm_ready()
    t0 = time.perf_counter()
    result = GRAPH.invoke({
        "question": question,
        "filter_rel_dir": filter_rel_dir,
        "filter_file_type": filter_file_type,
        "use_llm": use_llm,
    })
    total = round(time.perf_counter() - t0, 3)
    trace = result.get("trace", {})
    trace["latency"] = total

    answer = result.get("answer")
    sources = result.get("graded", [])
    # 生产级可观测: 线上真实问答的 LLM 忠实度自评(可选)
    if OBS.get("live_judge") and use_llm and answer:
        try:
            from .eval import live_faithfulness
            ctx = "\n".join(d.get("content", "") for d in sources)
            fscore = live_faithfulness(question, answer, ctx)
            if fscore is not None:
                trace["faithfulness"] = fscore
        except Exception as e:
            LOG.warning("线上自评失败(不影响回答): %s", e)

    # 落盘可观测记录
    obs.record({
        "question": question,
        "answer": (answer or "")[:500],
        "note": result.get("note", ""),
        "n_sources": len(sources),
        "sources_files": [s.get("file_name", "") for s in sources[:3]],
        **trace,
    })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "note": result.get("note", ""),
        "trace": trace,
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "请假流程怎么走"
    out = run_rag(q)
    print("问题:", q)
    print("答案:", out["answer"] or out["note"])
    print("来源数:", len(out["sources"]))
    for i, s in enumerate(out["sources"][:3], 1):
        print(f"  [{i}] {s.get('file_name')} | {s.get('rel_dir','')[:30]} "
              f"| rerank={s.get('rerank_score',0):.3f}")
