"""
配置加载模块
- 读取 config.yaml
- 合并 .env (LLM_API_KEY 等敏感信息)
- 解析为全局可用的 CONFIG 对象
- data 下路径若为相对路径, 则以项目根目录为基准解析(可整体搬迁)
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
CONFIG_EXAMPLE = BASE_DIR / "config.example.yaml"
ENV_PATH = BASE_DIR / ".env"

# === 关键: 把 HuggingFace 缓存/下载强制锁在项目内的 E 盘目录 ===
# 之前 C 盘被 HF 模型缓存撑爆, 这里改为写入项目 .hf_cache, 并走国内镜像加速下载。
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".hf_cache"))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _resolve(p: str) -> str:
    """相对路径以项目根目录解析, 绝对路径原样归一化"""
    p = str(p)
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(str(BASE_DIR), p))


def load_config():
    load_dotenv(ENV_PATH)
    # 优先用本地 config.yaml; 不存在时(如全新 clone)回退到示例配置
    src = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 路径归一化 / 相对路径解析
    data = cfg["data"]
    for k in ("source_dir", "chroma_dir", "sqlite_path", "ocr_cache_dir", "bm25_cache"):
        data[k] = _resolve(data[k])

    # LLM: 环境变量覆盖
    llm = cfg["llm"]
    env_key = os.getenv("LLM_API_KEY")
    if env_key:
        llm["api_key"] = env_key
    env_base = os.getenv("LLM_API_BASE")
    if env_base:
        llm["api_base"] = env_base
    env_model = os.getenv("LLM_MODEL")
    if env_model:
        llm["model"] = env_model

    # 创建必要目录
    for k in ("chroma_dir", "ocr_cache_dir"):
        os.makedirs(data[k], exist_ok=True)
    os.makedirs(os.path.dirname(data["sqlite_path"]), exist_ok=True)

    return cfg


CONFIG = load_config()
DATA = CONFIG["data"]
EMB = CONFIG["embedding"]
RERANK = CONFIG["rerank"]
RET = CONFIG["retrieval"]
LLM = CONFIG["llm"]
SERVER = CONFIG["server"]
OBS = CONFIG.get("observability", {})
