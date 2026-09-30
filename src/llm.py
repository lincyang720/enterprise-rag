"""
LLM 答案生成 (OpenAI 兼容接口)
- 支持 DeepSeek / 通义千问 / OpenAI / 本地 Ollama 等
- 无 Key 或关闭时返回 None, 由上层降级为仅展示检索片段
"""
from .config import LLM
from .utils import get_logger

LOG = get_logger()


def build_context(passages: list, per_passage_cap: int = 1500,
                  total_budget: int = 6000) -> str:
    """把检索到的片段拼成送给大模型的上下文。

    - 每条来源最多取 per_passage_cap 字, 保证单条片段完整可读;
    - 累计达到 total_budget 后停止, 避免超出模型上下文窗口 / 浪费 token。
    """
    lines = []
    total = 0
    for i, p in enumerate(passages, 1):
        head = f"[{i}] 文件: {p.get('file_name','')} (路径: {p.get('rel_dir','')})"
        body = p.get("content", "")[:per_passage_cap]
        total += len(body)
        lines.append(f"{head}\n{body}")
        if total >= total_budget:
            break
    return "\n\n".join(lines)


def answer_with_llm(query: str, passages: list):
    if not LLM.get("use_llm"):
        return None
    if not LLM.get("api_key"):
        LOG.warning("未配置 LLM_API_KEY, 仅返回检索片段")
        return None
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        LOG.warning("langchain-openai 未安装: %s", e)
        return None

    try:
        llm = ChatOpenAI(
            base_url=LLM["api_base"],
            api_key=LLM["api_key"],
            model=LLM["model"],
            temperature=LLM["temperature"],
            max_tokens=LLM["max_tokens"],
            timeout=60,
        )
        context = build_context(passages)
        human = (
            f"【参考资料】\n{context}\n\n"
            f"【用户问题】\n{query}\n\n"
            "请严格基于以上参考资料回答。"
        )
        resp = llm.invoke([
            SystemMessage(content=LLM["system_prompt"]),
            HumanMessage(content=human),
        ])
        return resp.content
    except Exception as e:
        LOG.error("LLM 调用失败: %s", e)
        return None
