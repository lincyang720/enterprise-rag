"""
可观测层 (Observability)
========================
生产级 RAG 必备: 对每一次问答做链路追踪与指标沉淀, 而不是只跑一次性评测。

提供:
- RunTrace: 记录单次问答的节点耗时、召回数、重排分数、Token、在线自评等。
- trace_store: 最近 N 条运行写入内存环形缓冲 + data/traces.jsonl(便于排查与复盘)。
- summarize(): 聚合延迟分位、平均召回、平均自评分数等运行期指标。

LLM-as-a-Judge 的离线/在线打分逻辑在 src/eval.py 与 src/rag_graph.py 中调用本层落盘。
"""
import os
import json
import time
from collections import deque

from .config import DATA
from .utils import get_logger

LOG = get_logger()

_TRACE_FILE = os.path.join(os.path.dirname(DATA["sqlite_path"]), "traces.jsonl")
_RECENT: deque = deque(maxlen=300)


def record(trace: dict):
    """落盘一条运行记录(内存环形缓冲 + JSONL)"""
    trace = dict(trace)
    trace.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    _RECENT.append(trace)
    try:
        with open(_TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    except Exception as e:
        LOG.warning("trace 落盘失败: %s", e)


def recent_traces(n: int = 50):
    return list(_RECENT)[-n:]


def summarize(n: int = 100):
    traces = list(_RECENT)[-n:]
    if not traces:
        return {"count": 0}
    lats = [t.get("latency", 0) for t in traces if t.get("latency")]
    recs = [t.get("retrieved", 0) for t in traces if t.get("retrieved")]
    top = [t.get("top_rerank", 0) for t in traces if t.get("top_rerank")]
    faiths = [t.get("faithfulness") for t in traces if t.get("faithfulness")]

    def pct(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
        return round(xs[k], 3)

    return {
        "count": len(traces),
        "latency_p50": pct(lats, 50),
        "latency_p95": pct(lats, 95),
        "avg_retrieved": round(sum(recs) / len(recs), 1) if recs else None,
        "avg_top_rerank": round(sum(top) / len(top), 3) if top else None,
        "avg_faithfulness": round(sum(faiths) / len(faiths), 3) if faiths else None,
        "with_judge": len(faiths),
    }
