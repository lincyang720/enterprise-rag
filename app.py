"""
Streamlit 局域网问答 + 评测界面
启动: streamlit run app.py --server.address=0.0.0.0 --server.port=8501

架构: 问答走 LangGraph 编排的 src/rag_graph.GRAPH (retrieve->rerank->grade->generate)
     评测走 src/eval (离线检索指标 + LLM-as-Judge 生成指标)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.config import SERVER, LLM
from src import store
from src import rag_graph
from src import eval as eval_mod
from src import observability as obs_mod
from src.llm import is_llm_ready
from src.utils import get_logger

st.set_page_config(page_title="bjhc 公司知识库问答", page_icon="📚", layout="wide")

# 侧边栏
st.sidebar.title("📚 bjhc 知识库")
st.sidebar.markdown("基于 RAG 的公司制度/项目资料问答")

try:
    n_chunks = store.count_chunks()
    n_files = store.count_files()
    st.sidebar.metric("已索引片段", n_chunks)
    st.sidebar.metric("已处理文件", n_files)
except Exception:
    st.sidebar.warning("索引库未初始化, 请先运行入库")

llm_on = is_llm_ready()
st.sidebar.markdown(
    f"**大模型答案**: {'✅ 已开启' if llm_on else '⚠️ 未配置(仅检索)'}")
st.sidebar.markdown("**检索**: BM25 + 向量余弦 + 重排序")
st.sidebar.markdown("**流程**: LangGraph(retrieve→rerank→grade→generate)")
st.sidebar.markdown(f"**服务地址**: `http://{SERVER['host']}:{SERVER['port']}`")

if st.sidebar.button("🔄 重建 BM25 缓存"):
    try:
        n = rag_graph.rebuild_bm25()
        st.sidebar.success(f"BM25 缓存已重建 (覆盖 {n} 个片段)")
    except Exception as e:
        st.sidebar.error(f"重建失败: {e}")
st.sidebar.caption("索引扩充后点此, 让关键词检索覆盖新入库内容。")

# 预热模型：把首次 embedding + reranker 加载放在页面打开时，而不是用户提问后
LOG = get_logger()
if not st.session_state.get("_models_warmed"):
    try:
        with st.spinner("🚀 首次启动正在加载 embedding + 重排序模型，约需 30–90 秒，请稍候..."):
            from src import embeddings as _emb, rerank as _rerank
            _emb.get_model()
            _rerank.get_model()
        st.session_state._models_warmed = True
        st.sidebar.success("模型加载完成，可以开始提问")
    except Exception as e:
        LOG.warning("模型预热失败: %s", e)
        st.sidebar.warning(f"模型预热失败，首次提问时仍会尝试加载：{e}")

# 过滤(仅问答页用)
try:
    conn = store.get_conn()
    dirs = [r[0] for r in conn.execute(
        "SELECT DISTINCT rel_dir FROM chunks ORDER BY rel_dir").fetchall()]
    ftypes = [r[0] for r in conn.execute(
        "SELECT DISTINCT file_type FROM chunks ORDER BY file_type").fetchall()]
except Exception:
    dirs, ftypes = [], []

TAB_QA, TAB_EVAL, TAB_OBS = st.tabs(["💬 智能问答", "📊 评测", "🔭 可观测"])

# ================= 问答页 =================
with TAB_QA:
    filter_dir = st.sidebar.selectbox("按目录过滤", ["(全部)"] + dirs)
    filter_ft = st.sidebar.selectbox("按类型过滤", ["(全部)"] + ftypes)

    st.title("公司资料智能问答")
    st.caption("输入问题, 系统从已索引的制度与项目资料中检索并回答, 附带来源。")

    if "history" not in st.session_state:
        st.session_state.history = []

    q = st.chat_input("例如: 请假流程怎么走? / 桥梁管理系统用的什么架构?")

    if q:
        with st.spinner("检索中..."):
            rel_dir = None if filter_dir == "(全部)" else filter_dir
            ft = None if filter_ft == "(全部)" else filter_ft
            res = rag_graph.run_rag(q, filter_rel_dir=rel_dir,
                                    filter_file_type=ft, use_llm=llm_on)
        st.session_state.history.append(
            (q, res.get("answer"), res.get("note", ""), res.get("sources", [])))

    for qi, ans, note, passages in st.session_state.history:
        st.chat_message("user").write(qi)
        with st.chat_message("assistant"):
            if ans:
                st.markdown(ans)
            else:
                st.info(note or "当前未配置大模型, 仅展示检索到的相关资料片段:")
            st.markdown("**📎 参考资料来源**")
            if not passages:
                st.warning("未检索到相关资料。")
            for i, p in enumerate(passages, 1):
                score = p.get("rerank_score", p.get("fusion_score", 0))
                with st.expander(
                    f"[{i}] {p.get('file_name','')}  ({p.get('rel_dir','')})  "
                    f"评分:{score:.3f}"):
                    st.write(p.get("content", ""))

    st.markdown("---")
    st.caption("混合检索(BM25关键词 + 向量语义) → 交叉编码器重排序 → LangGraph 筛选 → 大模型生成答案。")

# ================= 评测页 =================
with TAB_EVAL:
    st.title("RAG 评测")
    st.caption("离线检索指标(无需大模型, 随时可跑) + 生成指标(需配置 LLM_API_KEY)。")
    col1, col2 = st.columns(2)
    with col1:
        n_retrieval = st.number_input("检索评测样本数", 50, 2000, 200, step=50,
                                     key="n_ret")
    with col2:
        gen_n = st.number_input("生成评测样本数", 5, 100, 30, step=5,
                               key="gen_n")

    if st.button("🚀 运行检索评测 (离线)", type="primary"):
        with st.spinner("构造评测集并评估中..."):
            items = eval_mod.build_eval_set(n=int(n_retrieval))
            rep = eval_mod.retrieval_eval(items)
        if "error" in rep:
            st.error(rep["error"])
        else:
            st.success(f"检索评测完成, 样本 {rep['samples']} 条")
            c1, c2 = st.columns(2)
            c1.metric("文件级 HitRate@10",
                      rep["file_level"]["hit_rate"])
            c1.metric("文件级 MRR@10", rep["file_level"]["mrr"])
            c1.metric("文件级 NDCG@10", rep["file_level"]["ndcg"])
            c2.metric("片段级 HitRate@10",
                      rep["chunk_level"]["hit_rate"])
            c2.metric("片段级 MRR@10", rep["chunk_level"]["mrr"])
            c2.metric("片段级 NDCG@10", rep["chunk_level"]["ndcg"])
            st.markdown("**各截断位指标**")
            st.json(rep["file_level"])
            low = sorted(rep["per_query"],
                         key=lambda x: x.get("ndcg@10", 1))[:10]
            st.markdown("**检索较弱的样例(文件级 NDCG@10 最低 10 条)**")
            st.table([{"问题": r["query"], "文件": r["file"],
                       "NDCG@10": round(r.get("ndcg@10", 0), 3)} for r in low])

    if st.button("🧪 运行生成评测 (需大模型)"):
        if not llm_on:
            st.warning("未配置 LLM_API_KEY, 无法运行生成评测。请在 .env 中配置后重试。")
        else:
            with st.spinner("逐条生成并 LLM 打分中(可能较慢)..."):
                items = eval_mod.build_eval_set(n=int(n_retrieval))
                rep = eval_mod.generation_eval(items, n=int(gen_n))
            if rep.get("skipped"):
                st.warning(rep["reason"])
            else:
                st.success(f"生成评测完成, 样本 {rep['samples']} 条")
                m = rep["metrics"]
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("忠实度", m.get("faithfulness"))
                cc2.metric("答案相关性", m.get("answer_relevancy"))
                cc3.metric("上下文相关性", m.get("context_relevancy"))
                cc4.metric("上下文召回", m.get("context_recall"))
                st.markdown("**逐条明细**")
                st.table([{k: d.get(k) for k in
                           ("query", "faithfulness",
                            "answer_relevancy", "context_relevancy")}
                          for d in rep["details"]])

    st.markdown("---")
    st.caption("评测说明: 检索评测用'片段首句→其所属文件为正例'的伪黄金集衡量混合检索质量;"
               "生成评测用 LLM-as-Judge 对忠实度/相关性/召回打分。报告也保存在 data/eval_report.json。")

# ================= 可观测页 =================
with TAB_OBS:
    st.title("RAG 可观测 (Observability)")
    st.caption("生产级 RAG 必备: 实时掌握线上问答的链路耗时、召回质量与答案可信度。")

    if st.button("🔄 刷新运行指标"):
        st.rerun()

    summary = obs_mod.summarize(100)
    if summary.get("count"):
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("样本数", summary["count"])
        s2.metric("延迟 P50(s)", summary.get("latency_p50"))
        s3.metric("延迟 P95(s)", summary.get("latency_p95"))
        s4.metric("平均召回数", summary.get("avg_retrieved"))
        s5.metric("平均 Top重排分", summary.get("avg_top_rerank"))
        if summary.get("avg_faithfulness") is not None:
            st.metric("平均线上忠实度(开启 live_judge 后)",
                      summary.get("avg_faithfulness"),
                      help="在 config.yaml 设 observability.live_judge=true 后, "
                           "每次线上问答会用 LLM 自评忠实度并落盘")
    else:
        st.info("暂无运行记录。在'智能问答'页提问后, 这里会出现实时指标。")

    st.markdown("**最近问答链路**")
    traces = obs_mod.recent_traces(30)
    if traces:
        rows = []
        for t in traces[::-1]:
            rows.append({
                "时间": t.get("ts", ""),
                "问题": (t.get("question") or "")[:30],
                "延迟(s)": t.get("latency"),
                "召回": t.get("retrieved"),
                "重排": t.get("reranked"),
                "Top分": t.get("top_rerank"),
                "线上忠实度": t.get("faithfulness", "—"),
                "说明": (t.get("note") or "")[:20],
            })
        st.table(rows)
    else:
        st.caption("（暂无记录）")

    st.markdown("**评测历史趋势**")
    hist = eval_mod.load_eval_history()
    if hist:
        import pandas as pd
        df = pd.DataFrame(hist)
        df["ts"] = df["ts"].astype(str).str[5:16]
        chart_cols = [c for c in
                     ["ret_file_ndcg", "ret_file_hit", "ret_file_mrr",
                      "gen_faithfulness", "gen_answer_rel",
                      "gen_ctx_rel", "gen_ctx_recall"]
                     if c in df.columns]
        if chart_cols:
            st.line_chart(df.set_index("ts")[chart_cols])
        st.caption("横轴为评测批次时间, 纵轴为各指标(0~1 或 1~5)。"
                   "便于观察检索/生成质量随索引扩充或参数调整后的变化。")
    else:
        st.caption("（运行评测后会出现趋势图）")
