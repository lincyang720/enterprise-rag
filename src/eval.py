"""
RAG 评测模块
============
两类评测:

1) 检索评测 (离线, 不需要大模型)
   - 从已索引语料中抽样构造"伪黄金集": 取每个片段首句作为问题, 其所属文件即正例。
   - 指标: HitRate@k / MRR@k / NDCG@k (文件级 + 片段级相关性)。
   - 衡量"混合检索(BM25+向量)本身"把对的东西找没找出来。

2) 生成评测 (需要大模型, 无 Key 自动跳过)
   - 对评测集逐条跑 run_rag 得到答案与上下文, 用 LLM-as-Judge 打分:
       * 忠实度 Faithfulness (有无编造/脱离上下文)
       * 答案相关性 Answer Relevancy (是否切题)
       * 上下文相关性 Context Relevancy (检索到的内容是否相关)
   - 各指标 1~5 分, 取均值。

结果写入 data/eval_report.json, 并可由 run_eval.py / app.py 调用展示。
"""
import os
import re
import json
import random
from typing import List, Dict, Any, Optional

import numpy as np

from .config import DATA, LLM
from .utils import get_logger
from . import store
from .retriever import HybridSearch

LOG = get_logger()

EVAL_SET_PATH = os.path.join(os.path.dirname(DATA["sqlite_path"]), "eval_set.json")
REPORT_PATH = os.path.join(os.path.dirname(DATA["sqlite_path"]), "eval_report.json")


# ---------------- 评测集构造 ----------------
def _first_sentence(text: str, max_len: int = 60) -> str:
    """从片段中取一句可作为"问题"的文本。

    xlsx 内容是"单元格用 | 拼接", 直接取首句会混入数值前缀。
    策略: 优先挑含中文且长度>=4 的单元格作为查询, 避免数字噪声。
    """
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) < 6:
            continue
        if not re.search(r"[\u4e00-\u9fff]", line):
            continue
        # 优先取 | 分隔出的含中文单元格
        parts = [p.strip() for p in re.split(r"\s*\|\s*", line)]
        cand = None
        for p in parts:
            if len(p) >= 4 and re.search(r"[\u4e00-\u9fff]", p):
                cand = p
                break
        if cand is None:
            cand = line
        for sep in ["。", "；", "！", "？", ".", ";"]:
            idx = cand.find(sep)
            if 0 < idx <= max_len:
                return cand[:idx].strip()
        return cand[:max_len].strip()
    return ""


def build_eval_set(n: int = 200, seed: int = 42,
                  min_chars: int = 50, force: bool = False) -> List[Dict]:
    """从已索引片段抽样构造伪黄金集; 已存在则直接复用(force=True 重建)"""
    if os.path.exists(EVAL_SET_PATH) and not force:
        with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    conn = store.get_conn()
    rows = conn.execute(
        "SELECT chunk_id, file_name, rel_dir, content FROM chunks "
        "WHERE length(content) >= ?", (min_chars,)).fetchall()
    if not rows:
        LOG.warning("索引为空, 无法构造评测集")
        return []
    rng = random.Random(seed)
    rng.shuffle(rows)
    items = []
    for cid, fname, rdir, content in rows[: max(n, 1)]:
        q = _first_sentence(content)
        if len(q) < 6:
            continue
        items.append({
            "query": q,
            "chunk_id": cid,
            "file_name": fname or "",
            "rel_dir": rdir or "",
        })
        if len(items) >= n:
            break
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    LOG.info("评测集已构造: %d 条 -> %s", len(items), EVAL_SET_PATH)
    return items


# ---------------- 检索评测 ----------------
def _metrics_at_k(rel_flags: List[int], ks: List[int]) -> Dict[str, float]:
    """rel_flags: 按排名顺序的相关性(0/1)列表; 返回各 k 的命中/MRR/NDCG"""
    rel = np.array(rel_flags, dtype=float)
    n = len(rel)
    gains = rel / np.log2(np.arange(2, n + 2))  # 位置 i(1-based) 的折损
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))) if len(ideal) else 0.0

    out = {}
    for k in ks:
        top = rel[:k]
        hit = 1.0 if top.sum() > 0 else 0.0
        # MRR
        pos = np.where(top == 1)[0]
        mrr = (1.0 / (pos[0] + 1)) if pos.size else 0.0
        # NDCG
        dcg_k = float(np.sum(gains[:k]))
        ndcg = (dcg_k / idcg) if idcg > 0 else 0.0
        out[f"hit@{k}"] = hit
        out[f"mrr@{k}"] = mrr
        out[f"ndcg@{k}"] = ndcg
    return out


def retrieval_eval(items: List[Dict], ks: List[int] = None) -> Dict[str, Any]:
    if not items:
        return {"error": "评测集为空"}
    ks = ks or [1, 3, 5, 10]
    hybrid = HybridSearch()
    top_k = max(ks) + 5
    file_hits, chunk_hits = [], []
    file_mrr, chunk_mrr = [], []
    file_ndcg, chunk_ndcg = [], []

    per_query = []
    for it in items:
        cands = hybrid.retrieve_candidates(it["query"], top_k=top_k)
        file_rel = [1 if c.get("file_name") == it["file_name"] else 0 for c in cands]
        chunk_rel = [1 if c.get("chunk_id") == it["chunk_id"] else 0 for c in cands]
        fm = _metrics_at_k(file_rel, ks)
        cm = _metrics_at_k(chunk_rel, ks)
        file_hits.append(fm["hit@%d" % ks[-1]])
        chunk_hits.append(cm["hit@%d" % ks[-1]])
        file_mrr.append(fm["mrr@%d" % ks[-1]])
        chunk_mrr.append(cm["mrr@%d" % ks[-1]])
        file_ndcg.append(fm["ndcg@%d" % ks[-1]])
        chunk_ndcg.append(cm["ndcg@%d" % ks[-1]])
        per_query.append({"query": it["query"], "file": it["file_name"],
                          "file_ndcg@%d" % ks[-1]: fm["ndcg@%d" % ks[-1]]})

    def agg(xs):
        return round(float(np.mean(xs)), 4) if xs else 0.0

    return {
        "type": "retrieval",
        "samples": len(items),
        "ks": ks,
        "file_level": {
            "hit_rate": agg(file_hits), "mrr": agg(file_mrr), "ndcg": agg(file_ndcg)},
        "chunk_level": {
            "hit_rate": agg(chunk_hits), "mrr": agg(chunk_mrr), "ndcg": agg(chunk_ndcg)},
        "per_query": per_query,
    }


# ---------------- 生成评测 (LLM-as-Judge) ----------------
_JUDGE_PROMPTS = {
    "faithfulness": (
        "你是一个严格的评测员。给定【用户问题】【参考资料】【模型答案】，"
        "请判断答案是否完全基于参考资料、没有编造或引入资料之外的信息。\n"
        "只输出 JSON: {\"score\": 1~5的整数, \"reason\": \"简短说明\"}\n"
        "5=完全忠实, 1=大量编造/脱离资料。"),
    "answer_relevancy": (
        "你是一个严格的评测员。给定【用户问题】【模型答案】，"
        "请判断答案是否切题、有用、直接回答了问题。\n"
        "只输出 JSON: {\"score\": 1~5的整数, \"reason\": \"简短说明\"}\n"
        "5=高度相关且完整, 1=答非所问。"),
    "context_relevancy": (
        "你是一个严格的评测员。给定【用户问题】【参考资料】，"
        "请判断参考资料是否与问题相关、能否支撑回答。\n"
        "只输出 JSON: {\"score\": 1~5的整数, \"reason\": \"简短说明\"}\n"
        "5=高度相关, 1=完全无关。"),
    "context_recall": (
        "你是一个严格的评测员。给定【用户问题】【参考资料】，"
        "请判断参考资料是否包含了回答问题所需的全部关键信息(检索召回是否充分)。\n"
        "只输出 JSON: {\"score\": 1~5的整数, \"reason\": \"简短说明\"}\n"
        "5=资料充分覆盖所需信息, 1=资料严重缺失关键信息。"),
}


def _make_judge():
    if not (LLM.get("use_llm") and LLM.get("api_key")):
        return None
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(
            base_url=LLM["api_base"], api_key=LLM["api_key"],
            model=LLM["model"], temperature=0.0, max_tokens=400, timeout=60,
            model_kwargs={"response_format": {"type": "json_object"}})
        return llm
    except Exception as e:
        LOG.warning("评测 Judge 模型不可用: %s", e)
        return None


def _judge_one(llm, metric: str, question: str, answer: str,
               context: str) -> Optional[Dict]:
    sys_msg = _JUDGE_PROMPTS[metric]
    human = (
        f"【用户问题】\n{question}\n\n"
        f"【参考资料】\n{context[:3000]}\n\n"
        f"【模型答案】\n{answer}\n\n"
        "请严格只输出 JSON。")
    try:
        resp = llm.invoke([SystemMessage(content=sys_msg),
                           HumanMessage(content=human)])
        txt = resp.content.strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = txt[txt.find("{") - 1:] if "{" in txt else txt
        start = txt.find("{")
        end = txt.rfind("}")
        obj = json.loads(txt[start:end + 1])
        return {"score": float(obj.get("score", 0)),
                "reason": str(obj.get("reason", ""))[:80]}
    except Exception as e:
        LOG.warning("Judge(%s) 解析失败: %s", metric, e)
        return None


def live_faithfulness(question: str, answer: str, context: str) -> Optional[float]:
    """线上单次问答的忠实度自评(供可观测层调用), 无 Key 返回 None"""
    llm = _make_judge()
    if llm is None:
        return None
    res = _judge_one(llm, "faithfulness", question, answer, context)
    return res["score"] if res else None


def generation_eval(items: List[Dict], n: int = 30) -> Dict[str, Any]:
    llm = _make_judge()
    if llm is None:
        return {"type": "generation", "skipped": True,
                "reason": "未配置 LLM_API_KEY, 跳过生成评测(仅离线检索评测可用)"}
    from .rag_graph import run_rag

    items = items[:n]
    dims = ["faithfulness", "answer_relevancy", "context_relevancy", "context_recall"]
    scores = {k: [] for k in dims}
    details = []
    for it in items:
        res = run_rag(it["query"], use_llm=True)
        ans = res.get("answer")
        ctx = "\n".join(d.get("content", "") for d in res.get("sources", []))
        if not ans:
            continue
        judged = {k: _judge_one(llm, k, it["query"], ans, ctx) for k in dims}
        row = {"query": it["query"]}
        for k in dims:
            if judged[k]:
                scores[k].append(judged[k]["score"])
                row[k] = judged[k]["score"]
        details.append(row)

    def avg(xs):
        return round(float(np.mean(xs)), 3) if xs else None

    return {
        "type": "generation",
        "samples": len(details),
        "metrics": {k: avg(v) for k, v in scores.items()},
        "details": details,
    }


# ---------------- 评测历史(趋势) ----------------
HISTORY_PATH = os.path.join(os.path.dirname(DATA["sqlite_path"]), "eval_history.json")


def _save_history_entry(entry: Dict):
    try:
        hist = []
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                hist = json.load(f)
        hist.append(entry)
        hist = hist[-50:]  # 保留最近 50 次
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG.warning("评测历史写入失败: %s", e)


def load_eval_history() -> List[Dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ---------------- 总入口 ----------------
def run_eval(mode: str = "all", n: int = 200, gen_n: int = 30,
             ks: List[int] = None, force_set: bool = False) -> Dict[str, Any]:
    ks = ks or [1, 3, 5, 10]
    items = build_eval_set(n=n, force=force_set)
    report: Dict[str, Any] = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "indexed_chunks": store.count_chunks(),
        "indexed_files": store.count_files(),
    }
    if mode in ("retrieval", "all"):
        report["retrieval"] = retrieval_eval(items, ks)
    if mode in ("generation", "all"):
        report["generation"] = generation_eval(items, n=gen_n)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # 评测历史趋势(只存聚合指标, 不存逐条细节)
    entry = {"ts": report.get("generated_at"),
             "indexed_chunks": report.get("indexed_chunks")}
    if "retrieval" in report and "error" not in report["retrieval"]:
        r = report["retrieval"]
        entry["ret_file_ndcg"] = r["file_level"]["ndcg"]
        entry["ret_file_hit"] = r["file_level"]["hit_rate"]
        entry["ret_file_mrr"] = r["file_level"]["mrr"]
    if "generation" in report and not report["generation"].get("skipped"):
        g = report["generation"]
        entry["gen_faithfulness"] = g["metrics"].get("faithfulness")
        entry["gen_answer_rel"] = g["metrics"].get("answer_relevancy")
        entry["gen_ctx_rel"] = g["metrics"].get("context_relevancy")
        entry["gen_ctx_recall"] = g["metrics"].get("context_recall")
    _save_history_entry(entry)
    LOG.info("评测报告已写入: %s", REPORT_PATH)
    return report


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    rep = run_eval(mode=m)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
