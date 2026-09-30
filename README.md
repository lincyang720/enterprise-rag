# 企业知识库 RAG 问答系统（LangChain + LangGraph）

一套**开箱即用的企业制度 / 项目资料智能问答系统**。基于通用 RAG 框架（LangChain + LangGraph）构建，支持：

- 混合检索（**BM25 关键词 + 向量语义**）加权融合 + 交叉编码器**重排序**
- **LangGraph** 编排的可追溯问答流程（retrieve → rerank → grade → generate）
- 内置**评测**（离线检索指标 + LLM-as-Judge 四维度）与**可观测**（链路追踪 / 延迟分位 / 评测趋势）
- **网页界面**（Streamlit），局域网内同事浏览器即可访问
- 数据默认**完全本地化**（Embedding/Reranker 跑在本地，模型与索引全落项目内，不写系统盘，可整体搬迁）

> 适合：把公司散落的规章制度、项目文档、图纸说明等统一成一个"能问答的知识库"，让同事用自然语言快速查资料。

---

## 技术栈

| 环节 | 选型 |
|------|------|
| 向量库 | Chroma（本地持久化，余弦相似度） |
| 关键词检索 | BM25（`rank_bm25` + `jieba` 中文分词） |
| 融合 | BM25 与向量检索 min-max 归一化后加权融合 |
| 重排序 | `BAAI/bge-reranker-base` 交叉编码器 |
| Embedding | `BAAI/bge-small-zh-v1.5`（本地，数据不出内网） |
| 答案生成 | OpenAI 兼容接口（DeepSeek / 通义千问 / OpenAI / 本地 Ollama），可选 |
| 流程编排 | **LangGraph** 状态图，编译为 LangChain Runnable，可接入任意 Agent 框架 |
| 评测 | 离线检索指标（HitRate/MRR/NDCG@k）+ LLM-as-Judge（忠实度/答案相关性/上下文相关性/上下文召回） |
| 可观测 | 每次问答链路追踪落盘，界面聚合延迟 P50/P95、召回、重排分、评测趋势 |
| 界面 | Streamlit（局域网 `0.0.0.0` 暴露，含「问答 / 评测 / 可观测」三标签页） |

---

## 目录结构

```
bjhc-rag/
├── config.example.yaml   # 配置模板(复制为 config.yaml 后修改) ★ 已纳入版本库
├── config.yaml           # 本地真实配置(含本机资料路径) ★ 不入库(.gitignore)
├── .env.example          # LLM 密钥模板(复制为 .env 后填值) ★ 已纳入版本库
├── .env                  # 本地密钥 ★ 不入库
├── requirements.txt      # 依赖
├── start.py              # 启动助手(读取 server 配置拉起 Streamlit)
├── start_app.bat         # 一键启动界面(Windows)
├── start_ingest.bat      # 一键文本入库(Windows)
├── start_ocr.bat         # 一键图片 OCR 补录(Windows)
├── run_ingest.py         # 入库入口
├── run_ocr.py            # 图片 OCR 补录入库入口
├── run_eval.py           # 评测命令行入口
├── app.py                # Streamlit 问答界面
├── src/
│   ├── config.py        # 配置加载 + HF_HOME 锁项目内 + 回退 config.example.yaml
│   ├── utils.py         # 日志/指纹/分词/清洗
│   ├── parsers.py       # 多格式解析(pdf/doc/docx/xlsx/xls/pptx/txt + 图纸解包 + 图片OCR)
│   ├── chunker.py       # 滑动窗口分块
│   ├── embeddings.py    # 本地向量化
│   ├── store.py         # SQLite(原文/指纹) + Chroma(向量)
│   ├── rerank.py        # 交叉编码器重排序
│   ├── retriever.py     # BM25+向量融合 + LangChain 封装(BaseRetriever)
│   ├── rag_graph.py     # LangGraph 编排的问答状态图(retrieve→rerank→grade→generate)
│   ├── ocr_ingest.py    # 图片 OCR 补录(断点续传)
│   ├── eval.py          # RAG 评测(离线检索指标 + LLM-as-Judge 生成指标)
│   ├── llm.py           # LLM 答案生成(含降级)
│   └── ingest.py        # 入库管线(断点续传, 自动跳过 .svn/.git 等内部目录)
└── data/                # 索引库(自动生成, ★ 不入库) chroma / chunks.db / ocr_cache / bm25.pkl / traces.jsonl / eval_*.json
```

> **哪些不入库**：`data/`（含公司资料索引）、`venv/`、`models/`、`.hf_cache/`（模型权重）、`logs/`、`config.yaml`（本机路径）、`.env`（密钥）、`__pycache__/` 等，见 `.gitignore`。仓库只包含**代码 + 配置模板 + 文档**。

---

## 一、从 GitHub 克隆后部署（首次）

```bat
git clone <你的仓库地址>
cd bjhc-rag

:: 1) 建虚拟环境并装依赖
python -m venv venv
venv\Scripts\activate            :: macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

:: 2) 生成本地配置
copy config.example.yaml config.yaml
::   编辑 config.yaml, 把 data.source_dir 改成你自己的资料根目录

:: 3) (可选但建议) 配置大模型密钥
copy .env.example .env
::   编辑 .env 填入 LLM_API_KEY(DeepSeek / 通义千问等便宜模型即可)
::   不配置则系统仅返回检索片段, 不做自然语言生成

:: 4) 入库
python -u run_ingest.py --no-images

:: 5) 启动界面
python -u start.py
```

同事在浏览器访问 `http://<本机局域网IP>:8501` 即可使用（IP 用 `ipconfig` 查，通常 `192.168.x.x`）。

> macOS / Linux 把上面 `venv\Scripts\` 换成 `venv/bin/` 即可，其余命令一致。

---

## 二、已有 venv 的快速开始（Windows）

如果已经配好 `venv/`，可直接用一键脚本：

- **入库**：双击 `start_ingest.bat`，或 `venv\Scripts\python.exe -u run_ingest.py --no-images`
- **启动界面**：双击 `start_app.bat`，或 `venv\Scripts\python.exe -u start.py`
- **图片 OCR 补录**：双击 `start_ocr.bat`，或 `venv\Scripts\python.exe -u run_ocr.py`

---

## 三、入库说明

`run_ingest.py --no-images` 会递归扫描 `source_dir`，并**自动跳过**：

- 配置中的 `exclude_dirs`（如示例里的 `日报` / `软件` 等你不想要的目录）
- **版本控制 / 依赖 / 缓存目录**：`.svn`、`.git`、`.hg`、`__pycache__`、`node_modules`、`dist`、`build` 等，以及所有以 `.` 开头的隐藏目录
- 软件/二进制/归档扩展名（`.exe/.dll/.zip/.rar/.7z/.tar/.gz` 等，避免脏 chunk 污染索引）
- Office 临时锁文件（`~$` 开头）、系统文件（`Thumbs.db`/`Desktop.ini`）

入库**支持断点续传**：按文件指纹判断，已 `done` 的文件自动跳过，中断后重跑可放心续传。

> 提示：通用归档文件（`.zip` 等）若当文本读取会生成海量垃圾片段，因此**直接排除、不做解包**。如需检索压缩包内文档，请先解压到资料目录。

---

## 四、图片 OCR 补录（补齐"全量含图片"）

主入库用 `--no-images` 跳过图片（OCR 依赖 easyocr 模型，首次需联网下载权重）。待网络/模型可用后单独补录：

```bat
venv\Scripts\python.exe -u run_ocr.py
```

- 已做文件级缓存 + 断点续传；OCR 失败的图片标记 `empty`，后续重跑即可补齐。
- 补录完成后，在界面侧边栏点 **「🔄 重建 BM25 缓存」**（或重启界面）让新内容生效。

---

## 五、评测与可观测（生产级 RAG 必备）

参考行业实践（约 89% 的生产级 RAG 团队都在做 Observability + LLM-as-Judge），本系统内置：

### 评测（量化 RAG 质量）
```bat
venv\Scripts\python.exe run_eval.py --mode retrieval --n 200      # 离线检索评测(无需大模型, 秒级)
venv\Scripts\python.exe run_eval.py --mode generation --gen-n 30  # 生成评测(需 LLM_API_KEY)
venv\Scripts\python.exe run_eval.py --mode all                    # 两者都跑
```
- **检索评测**：从已索引片段抽样，取片段首句作问题、所属文件为正例，计算文件级/片段级 HitRate@k、MRR@k、NDCG@k。
- **生成评测（LLM-as-Judge）**：对评测集跑问答得答案，用 LLM 从四维度打分（1~5）：
  - **忠实度 Faithfulness**：答案是否完全基于资料、无编造
  - **答案相关性 Answer Relevancy**：是否切题有用
  - **上下文相关性 Context Relevancy**：检索内容是否相关
  - **上下文召回 Context Recall**：资料是否覆盖回答所需关键信息
- 报告落地 `data/eval_report.json`，评测历史趋势落地 `data/eval_history.json`；Web「评测」页可点击运行并可视化。

### 可观测（Observability）
- 每次真实问答的链路追踪落盘 `data/traces.jsonl`：节点耗时（retrieve/rerank/grade/generate）、召回数、Top 重排分、来源文件，以及（开启 `live_judge` 后）线上答案的 LLM 忠实度自评。
- Web「🔭 可观测」页展示：运行期指标（延迟 P50/P95、平均召回、平均 Top 重排分）、最近问答链路、评测历史趋势。
- `config.yaml` 设 `observability.live_judge: true` 开启线上逐条忠实度自评（会多一次大模型调用）。

---

## 六、查看入库进度（无需看日志）

入库是耗时操作（文件数依资料规模而定，常见数千到数万）。随时在另一窗口运行：
```bat
venv\Scripts\python.exe -c "import sqlite3;c=sqlite3.connect('data/chunks.db');print('片段:',c.execute('select count(*) from chunks').fetchone()[0]);[print(s,n) for s,n in c.execute('select status,count(*) from files group by status')]"
```
- `done` 数量持续增长即正常；`empty` 多为无文本的图片/二进制；`error` 为极少数解析失败文件，不影响整体。
- 入库过程中界面也能打开，但只检索已入库部分；**建议入库完成后再正式给同事使用**，或检索后点侧边栏「🔄 重建 BM25 缓存」。

---

## 七、检索与流程原理

1. **解析分块**：多格式文档解析为文本，滑动窗口切分为带元数据的 chunk。
2. **双路召回**：向量语义检索（Chroma 余弦）+ 关键词检索（BM25 / jieba）。
3. **融合**：两路分数 min-max 归一化后加权求和，取 top-N 候选。
4. **重排序**：cross-encoder 对候选重新打分，精排 top-K。
5. **LangGraph 编排**：`retrieve→rerank→grade(相关性筛选, 条件边)→generate`；无相关文档走 `no_answer`。图编译为 Runnable，可直接接入 Agent。
6. **生成**：精排片段作为上下文送 LLM，要求严格基于资料、标注来源，未找到则明说。

---

## 八、配置说明（config.yaml）

- `data.source_dir`：你的资料根目录（必填）。
- `data.exclude_dirs` / `exclude_exts` / `code_exts`：排除规则。
- `embedding.model_name` / `rerank.model_name`：可换成其他中文模型。
- `retrieval.chunk_size` / `bm25_weight` / `vector_weight`：分块与融合权重。
- `llm.*`：大模型接口；`use_llm: false` 可关闭生成仅做检索。
- `server.host` / `port`：界面暴露地址（默认 `0.0.0.0:8501`，即局域网可访问）。
- `observability.*`：`live_judge` 与 `keep_traces`。

---

## 九、注意事项

- 首次运行会自动从 HuggingFace 镜像（hf-mirror）下载 embedding 与 rerank 模型，**需联网一次**。
- **模型缓存与索引数据全部落在项目内**（`.hf_cache/`、`data/`），不会写入系统 C 盘；`config.py` 已强制把 `HF_HOME` 指向项目内目录，避免撑爆系统盘。
- 索引数据随项目整体复制迁移；迁移后重跑入库即可重建 BM25 缓存。
- 图片 OCR 使用 easyocr，首次加载模型需联网；已做文件级缓存避免重复识别。
- 敏感资料建议用本地 Ollama 等内网模型，确保数据不出企业网络。
- 本项目不收集、不上传任何资料内容；所有检索与生成均在本地完成（除非你主动配置公网 LLM 接口）。

---

## License

[MIT](LICENSE)
