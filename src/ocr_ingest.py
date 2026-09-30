"""
图片 OCR 补录脚本 (src/ocr_ingest.py)
==================================
用途: 主入库常用 --no-images 跳过图片(因为 OCR 依赖 easyocr 模型且首次需联网下载),
      本脚本在主入库完成后, 单独把 jpg/png/jpeg 图片 OCR 进库, 补齐"全量含图片"目标。

特性:
- 复用断点续传(文件指纹): 已 done 的图片跳过, 重跑安全。
- easyocr 模型首次需联网下载; 若当前环境无法下载, parse_image 返回 "" -> 标记 empty,
  待网络/模型可用后重跑本脚本即可补齐, 无需改动其它代码。
- OCR 文本按文件指纹缓存在 data/ocr_cache, 不会重复识别同一张图。

用法:
  python run_ocr.py            # 全量图片 OCR 补录
  python run_ocr.py --limit 20 # 调试: 仅前 20 张
  python run_ocr.py --reset    # 清空图片相关状态后重来(一般不必要)
"""
import os
import time
import argparse

from .config import DATA, RET
from .utils import get_logger, file_fingerprint, stable_id
from . import parsers, chunker, embeddings, store

LOG = get_logger()
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def iter_images(root, exclude_dirs):
    for dirpath, dirnames, filenames in os.walk(root):
        # 剪枝: 不进入排除目录
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            if fn.startswith("~$") or fn in ("Thumbs.db", "Desktop.ini", ".DS_Store"):
                continue
            fp = os.path.normpath(os.path.join(dirpath, fn))
            if os.path.splitext(fp)[1].lower() in IMAGE_EXTS:
                yield fp


def process_image(fp, root):
    mtime, size = file_fingerprint(fp)
    st = store.get_file_status(fp)
    if st and st[2] == "done" and st[0] == mtime and st[1] == size:
        return 0, "skipped"
    # 图片 OCR(无模型时返回 "", 标记 empty, 后续可重跑)
    text = parsers.parse_image(fp, DATA["ocr_cache_dir"])
    if not text or len(text) < 5:
        store.set_file_status(fp, mtime, size, "empty")
        return 0, "empty"
    chunks = chunker.chunk_text(text, RET["chunk_size"], RET["chunk_overlap"])
    if not chunks:
        store.set_file_status(fp, mtime, size, "empty")
        return 0, "empty"
    doc_id = stable_id(fp)
    embs = embeddings.embed_documents(chunks)
    records = []
    for i, ch in enumerate(chunks):
        ext = os.path.splitext(fp)[1].lower()
        records.append({
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "file_path": fp,
            "file_name": os.path.basename(fp),
            "file_type": ext,
            "rel_dir": os.path.dirname(os.path.relpath(fp, root)),
            "chunk_index": i,
            "content": ch,
            "embedding": embs[i],
        })
    store.insert_chunks(records)
    store.set_file_status(fp, mtime, size, "done", chunk_count=len(chunks))
    return len(chunks), "done"


def run(reset=False, limit=None):
    root = DATA["source_dir"]
    exclude_dirs = set(DATA["exclude_dirs"])
    LOG.info("开始扫描图片: %s (排除目录: %s)", root, exclude_dirs)
    files = list(iter_images(root, exclude_dirs))
    total = len(files)
    LOG.info("待 OCR 图片数: %d", total)

    stats = {"done": 0, "skipped": 0, "empty": 0, "error": 0}
    chunks_total = 0
    start = time.time()
    for idx, fp in enumerate(files, 1):
        if limit and idx > limit:
            break
        try:
            n, status = process_image(fp, root)
        except Exception as e:
            LOG.error("图片处理异常 %s: %s", fp, e)
            stats["error"] += 1
            continue
        stats[status] = stats.get(status, 0) + 1
        chunks_total += n
        if idx % 25 == 0 or idx == total:
            el = time.time() - start
            LOG.info("[%d/%d] 已处理图片, 累计片段 %d, 耗时 %.1fs",
                     idx, total, chunks_total, el)
    LOG.info("==== OCR 补录完成 ==== 图片:%d 片段:%d 统计:%s", total, chunks_total, stats)
    LOG.info("提示: 图片入库后请重启界面(或点侧边栏'重建 BM25 缓存')以生效。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="清空图片相关状态后重来")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 张(调试)")
    args = ap.parse_args()
    run(reset=args.reset, limit=args.limit)


if __name__ == "__main__":
    main()
