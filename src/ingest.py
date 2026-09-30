"""
入库管线
- 递归遍历 source_dir, 跳过 日报/软件 目录与软件/代码/二进制扩展名
- 解析 -> 分块 -> 本地向量化 -> 写入 sqlite + chroma
- 支持断点续传(文件指纹比对)与失败跳过
用法:
  python run_ingest.py            # 全量(含图片OCR)
  python run_ingest.py --no-images # 仅文本类(快速验证)
  python run_ingest.py --reset     # 清空后重来
"""
import os
import time
import argparse

from .config import DATA, RET
from .utils import file_fingerprint, stable_id, get_logger
from . import parsers, chunker, embeddings, store

LOG = get_logger()

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# 额外排除: 版本控制 / IDE / 依赖 / 构建缓存目录, 避免把 .svn/.git/__pycache__/node_modules
# 等内部文件当成公司文档入库(曾导致 96% 索引被 SVN 元数据污染)
VCS_DEP_DIRS = {
    ".svn", ".git", ".hg", ".idea", ".vscode",
    "__pycache__", "node_modules", "dist", "build", "bin", "obj", ".mypy_cache",
}


def _skip_set(no_images: bool):
    exts = set(DATA["exclude_exts"]) | set(DATA["code_exts"])
    if no_images:
        exts |= IMAGE_EXTS
    return exts


def iter_files(root, skip_exts, exclude_dirs):
    exclude = set(exclude_dirs) | VCS_DEP_DIRS
    for dirpath, dirnames, filenames in os.walk(root):
        # 剪枝: 不进入排除目录 / 隐藏目录(以 . 开头) / 版本控制与依赖目录
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude and not d.startswith(".")
        ]
        for fn in filenames:
            # 跳过 Office 临时锁文件 / 系统垃圾 / 隐藏文件(.gitignore/.env 等)
            if fn.startswith("~$") or fn.startswith(".") or fn in ("Thumbs.db", "Desktop.ini"):
                continue
            fp = os.path.normpath(os.path.join(dirpath, fn))
            ext = os.path.splitext(fp)[1].lower()
            if ext in skip_exts:
                continue
            yield fp


def process_file(fp, root, skip_if_done=True):
    mtime, size = file_fingerprint(fp)
    if skip_if_done:
        st = store.get_file_status(fp)
        if st and st[2] == "done" and st[0] == mtime and st[1] == size:
            return 0, "skipped"
    rel = os.path.relpath(fp, root)
    rel_dir = os.path.dirname(rel)
    ext = os.path.splitext(fp)[1].lower()
    file_name = os.path.basename(fp)

    try:
        text = parsers.parse_file(fp, DATA["ocr_cache_dir"])
    except Exception as e:
        store.set_file_status(fp, mtime, size, "error", error=str(e)[:200])
        return 0, "error"

    if not text or len(text) < 5:
        store.set_file_status(fp, mtime, size, "empty")
        return 0, "empty"

    chunks = chunker.chunk_text(text, RET["chunk_size"], RET["chunk_overlap"])
    if not chunks:
        store.set_file_status(fp, mtime, size, "empty")
        return 0, "empty"

    doc_id = stable_id(fp)
    # 批量向量化
    embs = embeddings.embed_documents(chunks)
    records = []
    for i, ch in enumerate(chunks):
        records.append({
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "file_path": fp,
            "file_name": file_name,
            "file_type": ext,
            "rel_dir": rel_dir,
            "chunk_index": i,
            "content": ch,
            "embedding": embs[i],
        })
    store.insert_chunks(records)
    store.set_file_status(fp, mtime, size, "done", chunk_count=len(chunks))
    return len(chunks), "done"


def run(reset=False, no_images=False, limit=None):
    root = DATA["source_dir"]
    if reset:
        LOG.info("清空旧索引...")
        try:
            store.get_chroma().delete(where={})
        except Exception:
            pass
        try:
            os.remove(DATA["sqlite_path"])
        except Exception:
            pass
        try:
            os.remove(DATA["bm25_cache"])
        except Exception:
            pass
        # 真正重置 store 模块内的连接缓存
        store._conn = None
        store._chroma = None

    skip_exts = _skip_set(no_images)
    exclude_dirs = set(DATA["exclude_dirs"])
    LOG.info("开始扫描: %s (排除目录: %s)", root, exclude_dirs)

    files = list(iter_files(root, skip_exts, exclude_dirs))
    total = len(files)
    LOG.info("待处理文件数: %d", total)

    stats = {"done": 0, "skipped": 0, "empty": 0, "error": 0}
    chunks_total = 0
    start = time.time()
    for idx, fp in enumerate(files, 1):
        if limit and idx > limit:
            break
        try:
            n, status = process_file(fp, root)
        except Exception as e:
            LOG.error("处理异常 %s: %s", fp, e)
            stats["error"] += 1
            continue
        stats[status] = stats.get(status, 0) + 1
        chunks_total += n
        if idx % 25 == 0 or idx == total:
            el = time.time() - start
            LOG.info("[%d/%d] 已处理, 累计片段 %d, 耗时 %.1fs", idx, total, chunks_total, el)

    el = time.time() - start
    LOG.info("==== 入库完成 ==== 文件:%d 片段:%d 耗时:%.1fs", total, chunks_total, el)
    LOG.info("统计: %s", stats)
    LOG.info("提示: 启动界面前请运行一次检索以构建 BM25 缓存(自动)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="清空后重来")
    ap.add_argument("--no-images", action="store_true", help="跳过图片OCR(快速)")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 个文件(调试)")
    args = ap.parse_args()
    run(reset=args.reset, no_images=args.no_images, limit=args.limit)


if __name__ == "__main__":
    main()
