"""
多格式文档解析器
支持: pdf / docx / doc / xlsx / xls / pptx / txt / csv / md / xml / html
      + 图纸类(eddx/emmx/vsdx/xmind/drawio/bpmn) zip 解包提取文本
      + 图片(jpg/png/jpeg) easyocr 识别
解析失败返回 ("", meta) 不影响整体入库。
"""
import os
import re
import json
import zipfile

from .utils import LOG, clean_text, extract_printable, file_fingerprint

# 纯文本类(直接读)
TEXT_EXTS = {
    ".txt", ".csv", ".md", ".json", ".yml", ".yaml", ".ini", ".properties",
    ".conf", ".log", ".sql", ".xml", ".html", ".htm", ".css",
}
# 需要 zip 解包的图纸/脑图类
ZIP_DIAGRAM_EXTS = {
    ".eddx", ".emmx", ".vsdx", ".xmind", ".drawio", ".bpmn", ".svg",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OFFICE_TEXT_EXTS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx"}

# ---------------- 文本读取 ----------------
def read_text_file(path: str) -> str:
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            break
    return ""

# ---------------- PDF ----------------
def parse_pdf(path: str) -> str:
    try:
        import fitz  # pymupdf
    except Exception as e:
        LOG.warning("pymupdf 未安装, 跳过 %s: %s", path, e)
        return ""
    try:
        doc = fitz.open(path)
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        doc.close()
        return clean_text("\n".join(parts))
    except Exception as e:
        LOG.warning("PDF 解析失败 %s: %s", path, e)
        return ""

# ---------------- DOCX ----------------
def parse_docx(path: str) -> str:
    try:
        from docx import Document
    except Exception as e:
        LOG.warning("python-docx 未安装, 跳过 %s", path)
        return ""
    try:
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return clean_text("\n".join(parts))
    except Exception as e:
        LOG.warning("DOCX 解析失败 %s: %s", path, e)
        return ""

# ---------------- DOC (老格式, olefile 尽力提取) ----------------
def parse_doc(path: str) -> str:
    try:
        import olefile
    except Exception:
        return ""
    try:
        ole = olefile.OleFileIO(path)
        texts = []
        for stream in ole.listdir():
            try:
                data = ole.openstream(stream).read()
            except Exception:
                continue
            for enc in ("gbk", "utf-8", "latin-1"):
                try:
                    t = data.decode(enc, errors="strict")
                    texts.append(t)
                    break
                except Exception:
                    continue
        ole.close()
        raw = "\n".join(texts)
        return extract_printable(raw)
    except Exception as e:
        LOG.warning("DOC 解析失败 %s: %s", path, e)
        return ""

# ---------------- XLSX ----------------
def parse_xlsx(path: str) -> str:
    try:
        import openpyxl
    except Exception as e:
        LOG.warning("openpyxl 未安装, 跳过 %s", path)
        return ""
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"【工作表: {ws.title}】")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        wb.close()
        return clean_text("\n".join(parts))
    except Exception as e:
        LOG.warning("XLSX 解析失败 %s: %s", path, e)
        return ""

# ---------------- XLS (老格式) ----------------
def parse_xls(path: str) -> str:
    try:
        import xlrd
    except Exception as e:
        LOG.warning("xlrd 未安装, 跳过 %s", path)
        return ""
    try:
        wb = xlrd.open_workbook(path)
        parts = []
        for ws in wb.sheets():
            parts.append(f"【工作表: {ws.name}】")
            for r in range(ws.nrows):
                cells = [str(c).strip() for c in ws.row_values(r) if str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return clean_text("\n".join(parts))
    except Exception as e:
        LOG.warning("XLS 解析失败 %s: %s", path, e)
        return ""

# ---------------- PPTX ----------------
def parse_pptx(path: str) -> str:
    try:
        from pptx import Presentation
    except Exception as e:
        LOG.warning("python-pptx 未安装, 跳过 %s", path)
        return ""
    try:
        prs = Presentation(path)
        parts = []
        for i, slide in enumerate(prs.slides, 1):
            parts.append(f"【幻灯片 {i}】")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    txt = shape.text_frame.text.strip()
                    if txt:
                        parts.append(txt)
                if shape.has_table:
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            parts.append(" | ".join(cells))
        return clean_text("\n".join(parts))
    except Exception as e:
        LOG.warning("PPTX 解析失败 %s: %s", path, e)
        return ""

# ---------------- ZIP 图纸/脑图 解包 ----------------
def _json_strings(obj):
    out = []
    if isinstance(obj, str):
        if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", obj):
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_json_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_json_strings(v))
    return out

def _xml_text(s: str) -> str:
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    attrs = re.findall(
        r'(?:name|title|label|text|value|description|content|topic|id|name_)="([^"]*)"',
        s)
    elems = re.findall(r">([^<>]+)<", s)
    cands = [a.strip() for a in attrs if re.search(r"[\u4e00-\u9fff]", a)]
    cands += [e.strip() for e in elems if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", e)]
    # 过滤掉纯标签名
    cands = [c for c in cands if len(c) >= 1]
    return "\n".join(cands)

def parse_zip_diagram(path: str) -> str:
    try:
        zf = zipfile.ZipFile(path)
    except Exception as e:
        LOG.warning("ZIP 解包失败 %s: %s", path, e)
        return ""
    parts = []
    try:
        for name in zf.namelist():
            try:
                data = zf.read(name)
            except Exception:
                continue
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if name.endswith(".json"):
                try:
                    obj = json.loads(text)
                    parts.append("\n".join(_json_strings(obj)))
                except Exception:
                    parts.append(extract_printable(text))
            elif name.endswith((".xml", ".svg", ".bpmn", ".drawio", ".edx", ".emmx")):
                parts.append(_xml_text(text))
            else:
                # 通用: 仅保留可打印片段
                parts.append(extract_printable(text))
    except Exception as e:
        LOG.warning("图纸解析异常 %s: %s", path, e)
    finally:
        zf.close()
    return clean_text("\n".join(parts))

# ---------------- 图片 OCR ----------------
_OCR_READER = None

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            from easyocr import Reader
            _OCR_READER = Reader(["ch_sim", "en"], gpu=False, verbose=False)
            LOG.info("easyocr 读者已加载")
        except Exception as e:
            LOG.error("easyocr 初始化失败: %s", e)
            _OCR_READER = False
    return _OCR_READER if _OCR_READER else None

def parse_image(path: str, ocr_cache_dir: str = None) -> str:
    reader = _get_ocr_reader()
    if reader is None:
        return ""
    # 文件缓存, 避免重复 OCR
    if ocr_cache_dir:
        fp = file_fingerprint(path)
        cache_name = f"{fp[0]}_{fp[1]}.txt"
        cache_path = os.path.join(ocr_cache_dir, cache_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    try:
        result = reader.readtext(path, detail=0, paragraph=True)
        text = clean_text("\n".join(result))
        if ocr_cache_dir:
            try:
                os.makedirs(ocr_cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
        return text
    except Exception as e:
        LOG.warning("OCR 失败 %s: %s", path, e)
        return ""

# ---------------- 统一入口 ----------------
def parse_file(path: str, ocr_cache_dir: str = None) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        return clean_text(read_text_file(path))
    elif ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".docx":
        return parse_docx(path)
    elif ext == ".doc":
        return parse_doc(path)
    elif ext == ".xlsx":
        return parse_xlsx(path)
    elif ext == ".xls":
        return parse_xls(path)
    elif ext == ".pptx":
        return parse_pptx(path)
    elif ext in IMAGE_EXTS:
        return parse_image(path, ocr_cache_dir)
    elif ext in ZIP_DIAGRAM_EXTS:
        return parse_zip_diagram(path)
    else:
        # 兜底: 尝试按文本读取
        return clean_text(read_text_file(path))
