"""
存储层
- SQLite: 原文 chunks + 已处理文件指纹(断点续传)
- Chroma: 稠密向量(余弦空间), 支持 upsert / 查询
"""
import os
import time
import sqlite3

from .config import DATA
from .utils import get_logger

LOG = get_logger()

_conn = None
_chroma = None


def get_conn():
    global _conn
    if _conn is None:
        path = DATA["sqlite_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                doc_id TEXT,
                file_path TEXT,
                file_name TEXT,
                file_type TEXT,
                rel_dir TEXT,
                chunk_index INTEGER,
                content TEXT,
                char_count INTEGER
            )""")
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS files(
                file_path TEXT PRIMARY KEY,
                mtime REAL,
                size INTEGER,
                status TEXT,
                error TEXT,
                chunk_count INTEGER,
                updated_at TEXT
            )""")
        _conn.commit()
    return _conn


def get_chroma():
    global _chroma
    if _chroma is None:
        from chromadb import PersistentClient
        client = PersistentClient(path=DATA["chroma_dir"])
        _chroma = client.get_or_create_collection(
            "bjhc_kb", metadata={"hnsw:space": "cosine"})
    return _chroma


def insert_chunks(records: list, batch_size: int = 500):
    """records: [{chunk_id,doc_id,file_path,file_name,file_type,rel_dir,chunk_index,content,embedding}]
    分批写入: SQLite executemany + Chroma upsert, 避免超大单批触发底层批量上限。"""
    if not records:
        return
    conn = get_conn()
    col = get_chroma()
    B = max(1, batch_size)
    for i in range(0, len(records), B):
        grp = records[i:i + B]
        rows = [
            (r["chunk_id"], r["doc_id"], r["file_path"], r["file_name"],
             r["file_type"], r["rel_dir"], r["chunk_index"], r["content"],
             len(r["content"]))
            for r in grp
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id,doc_id,file_path,file_name,file_type,rel_dir,chunk_index,content,char_count)
               VALUES (?,?,?,?,?,?,?,?,?)""", rows)
        conn.commit()
        ids = [r["chunk_id"] for r in grp]
        embeds = [r["embedding"] for r in grp]
        metas = [{
            "file_name": r["file_name"] or "",
            "file_type": r["file_type"] or "",
            "rel_dir": r["rel_dir"] or "",
            "chunk_index": int(r["chunk_index"]),
            "doc_id": r["doc_id"] or "",
        } for r in grp]
        docs = [r["content"] for r in grp]
        col.upsert(ids=ids, embeddings=embeds, metadatas=metas, documents=docs)


def get_all_chunks():
    """返回 [(chunk_id, content)] 用于构建 BM25"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id, content FROM chunks").fetchall()
    return [(r[0], r[1]) for r in rows]


def count_chunks():
    conn = get_conn()
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def count_files():
    conn = get_conn()
    return conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]


def set_file_status(path, mtime, size, status, error="", chunk_count=0):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO files
           (file_path,mtime,size,status,error,chunk_count,updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (path, mtime, size, status, error, chunk_count,
         time.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()


def get_file_status(path):
    conn = get_conn()
    return conn.execute(
        "SELECT mtime,size,status FROM files WHERE file_path=?",
        (path,)).fetchone()
