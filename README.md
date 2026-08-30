# hybrid-retrieval-knowledge-base-system

一个基于 **LangGraph + 通义千问 + 混合检索** 的 RAG（检索增强生成）知识库问答系统。用户提问后，系统先从本地知识库中检索出最相关的段落，再把段落拼进 Prompt 交给大模型生成回答，让 AI 只依据你的资料回答、不凭空编造，并标注来源。

项目面向「投资理财」垂直场景，内置 **20 份**投资理财知识文档（覆盖基金、债券、黄金、REITs、养老、打新、资产配置等主题，txt / md / docx / pdf 四种格式），但整体架构与具体业务解耦，可替换为任意领域的文档。

## 功能特性

- **混合检索**: BM25 关键词检索（rank-bm25 + jieba 分词）与向量语义检索（FAISS + BGE）双路召回，经 RRF 倒数排名融合，兼顾专有名词精确匹配与语义理解
- **BGE 重排序**: 用 BGE-Reranker 交叉编码器对候选段落精细重排，捕捉 query 与段落的词级交互，比向量相似度更准
- **智能路由**: LLM 先判断问题类型，简单问答直接回答，专业问题才走检索链路
- **分级记忆**: 短期记忆（会话内最近 5 轮）+ 长期记忆（Elasticsearch 跨会话，可选启用）
- **效果评估**: 确定性指标（Hit Rate@K / MRR / 关键词覆盖）+ LLM-as-Judge 自动打分（OpenEvals）
- **多入口**: 命令行、Web 网页、评估脚本三种使用方式
- **多格式文档**: 支持 txt / md / pdf / docx 四种文档格式

## 项目结构

```
hybrid-retrieval-knowledge-base-system/
├── app.py                    # Web 聊天界面入口（Flask）
├── main.py                   # 命令行交互入口
├── run_rag_evaluation.py     # 效果评估入口（Hit Rate / MRR / OpenEvals）
├── requirements.txt          # Python 依赖
├── set_env.bat / .ps1        # 配置 HuggingFace 缓存目录 + 国内镜像
├── setup_es.bat / .ps1       # 一键下载并启动本地 Elasticsearch
│
├── config/
│   └── settings.py           # 全局配置（模型、检索参数、路径）
│
├── core/                     # 核心编排
│   ├── state.py              # 状态数据结构定义
│   └── workflow.py           # LangGraph 工作流（7 个节点）
│
├── retrieval/                # 检索链路
│   ├── doc_parser.py         # 文档解析 + 分块
│   ├── bm25_retriever.py     # BM25 关键词检索
│   ├── vector_retriever.py   # 向量语义检索（FAISS）
│   ├── hybrid_retriever.py   # RRF 混合融合
│   └── reranker.py           # BGE 交叉编码器重排序
│
├── memory/                   # 分级记忆
│   ├── short_term.py         # 短期记忆（进程内）
│   └── long_term.py          # 长期记忆（Elasticsearch）
│
├── evaluation/               # 效果评估
│   ├── evaluator.py          # 评估器（Hit Rate / MRR / OpenEvals 裁判）
│   ├── metrics.py            # 确定性指标计算
│   └── golden_dataset.json   # 标准测试集（60 道题）
│
├── scripts/                  # 辅助脚本
│   ├── build_knowledge_docs.py   # 一键生成 20 份知识文档
│   ├── verify_golden_dataset.py  # 校验测试集与知识库一致性
│   └── tune_retrieval_params.py  # 检索参数网格搜索
│
├── data/
│   └── knowledge_base/       # 原始知识文档（.txt/.md/.docx/.pdf）
│
└── test/                     # 单元测试（unittest）
```

> `data/chunks/`、`data/vector_store/`、`data/elasticsearch-8.15.0/` 为运行时自动生成的缓存与索引，已被 `.gitignore` 忽略，无需手动管理。

## 快速开始

### 环境要求

- Python 3.9 及以上（建议 3.10+，本机验证 3.12）
- Windows（配套脚本为 `.bat` / `.ps1`；macOS / Linux 可直接运行 Python 入口）
- 网络：需访问阿里云 DashScope API；模型首次下载走国内镜像 `hf-mirror.com`
- （可选）Elasticsearch：仅长期记忆需要

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 HuggingFace 缓存与镜像（推荐）

双击 `set_env.bat`（或运行 `.\set_env.ps1`），完成两件事：

1. 把 HuggingFace 缓存目录指到 D 盘，避免占用 C 盘空间；
2. 配置国内镜像 `HF_ENDPOINT=https://hf-mirror.com`，让模型通过国内镜像下载。

> 首次运行会加载两个 BGE 模型：向量模型 `BAAI/bge-large-zh-v1.5`（约 1.3GB）与重排模型 `BAAI/bge-reranker-large`（约 2.2GB）。

### 3. 配置 API Key

系统通过**环境变量**读取密钥（未使用 `.env` 文件）。运行前先设置 `DASHSCOPE_API_KEY`：

**PowerShell：**

```powershell
$env:DASHSCOPE_API_KEY = "sk-你的通义千问密钥"
```

**CMD：**

```bat
set DASHSCOPE_API_KEY=sk-你的通义千问密钥
```

> 也可以把该变量配置到「系统环境变量」里，一劳永逸。

### 4. 准备知识库文档

把要让 AI 学习的文档放进 `data/knowledge_base/` 目录即可。系统支持 `.txt`、`.md`、`.pdf`、`.docx`，首次运行会自动解析文档、分块并建立索引。

项目已内置 **20 份**投资理财知识文档，格式分布为 txt × 8、md × 4、docx × 4、pdf × 4，覆盖基金进阶、债券、货币基金、指数基金、复利、资产配置、黄金、REITs、养老、止盈止损、打新等主题。如需重新生成这些文档，运行 `python scripts/build_knowledge_docs.py`。

## 运行方式

### 1. Web GUI 模式（推荐）

启动带图形界面的 Web 服务，在浏览器中与系统对话：

```bash
python app.py
```

浏览器打开 `http://localhost:8080`（端口可用环境变量 `PORT` 修改）。

### 2. CLI 命令行模式

在终端中直接提问：

```bash
python main.py
```

输入问题回车即可，输入 `q` / `exit` 退出。

### 3. 效果评估模式

```bash
# 本地检索指标评估（Hit Rate@5 / MRR，无需调用 LLM）
python run_rag_evaluation.py

# 指定 Hit Rate 的 Top-K
python run_rag_evaluation.py --k 3

# 完整 RAG 评估（含 LLM 生成 + OpenEvals 裁判）
python run_rag_evaluation.py --full-rag

# 上报 LangSmith（需先配置 LANGSMITH_API_KEY）
python run_rag_evaluation.py --full-rag --langsmith
```

## 核心概念

### 工作流程

系统用 LangGraph 定义了一条 7 节点的流水线（见 `core/workflow.py`）：

1. **assess_query**：调用 LLM 判断问题类型（`simple_qa` / `knowledge_retrieval` / `comparative_analysis` / `procedural_guide`）与处理模式（`reactive` / `deliberative`）
2. **路由**：`reactive` 简单问答直接走 `direct_answer`；`deliberative` 需要查资料走检索链
3. **hybrid_search**：BM25 与向量两路分别检索 Top-20，用 RRF 融合成候选集
4. **rerank**：BGE 交叉编码器对候选集精细重排，只保留最相关的 Top-5
5. **inject_memory**：注入短期记忆（最近几轮对话）+ 长期记忆（ES 里相似历史问答与用户画像）
6. **generate_answer / direct_answer**：把检索段落 + 记忆上下文拼进 Prompt，交给 LLM 生成回答
7. **store_memory**：把本轮问答写入短期记忆，若 ES 可用再写入长期记忆

### 混合检索 + 重排序

- **BM25**（rank-bm25 + jieba）：关键词精确匹配，专有名词（如 ETF、可转债）很准，但不懂同义词
- **向量检索**（FAISS + BGE）：语义相似度，能理解同义表达，但专有名词可能不准
- **RRF 融合**：两路结果按倒数排名融合成候选集，兼顾两者优点
- **BGE 交叉编码器**：把 query 与候选段落拼成一对联合打分，捕捉词级交互，精度最高但较慢，因此只对融合后的少量候选做精排

### 分级记忆

- **短期记忆**：进程内保存最近 5 轮对话，用于追问理解与上下文连贯
- **长期记忆**：Elasticsearch 跨会话检索相似历史问答与用户偏好，让回答更个性化（可选，不启动时自动跳过）

## 参与开发

### 代码约定

- 注释使用中文，UTF-8 编码
- 模块职责单一：`retrieval/` 只做检索、`memory/` 只做记忆、`core/workflow.py` 负责编排
- 不引入多余依赖，优先复用已有模块

### 常见扩展点

| 想做什么 | 改哪里 |
| --- | --- |
| 更换向量 / 重排模型 | `config/settings.py` 中的 `EMBEDDING_MODEL_NAME` / `RERANKER_MODEL_NAME` |
| 增加新知识文档 | 直接往 `data/knowledge_base/` 放文件 |
| 调整分块 / 检索参数 | `config/settings.py`（`CHUNK_SIZE` / `RRF_K` / `*_TOP_K`） |
| 增加新的评估指标 | `evaluation/metrics.py` 加函数，再在 `evaluation/evaluator.py` 注册 |

### 运行测试

项目使用标准库 `unittest`：

```bash
python -m unittest discover -s test
```

## 常见问题

**Q: 为什么不用 ChatGPT 直接回答？**
A: 通用大模型不知道你公司/个人的内部资料。RAG 的价值是让 AI 只依据你提供的文档回答，并标注来源，不瞎编。

**Q: BM25 和向量检索有什么区别？**
A: BM25 是「关键词精确匹配」，搜「ETF」很准但不懂同义词；向量检索是「语义理解」，能理解「养老储蓄 ≈ 退休规划」，但专有名词可能不准。两者混合互补。

**Q: 必须启动 Elasticsearch 吗？**
A: 不必须。ES 只负责长期记忆，不启动时系统自动跳过，其余功能正常。需要时双击 `setup_es.bat` 一键下载并启动。

**Q: 首次运行模型下载慢或超时？**
A: 模型默认走国内镜像 `hf-mirror.com` 下载（由 `set_env.bat` 配置 `HF_ENDPOINT`），已缓存的模型直接本地加载。若仍连不上镜像，可先用 `curl` / `git` 手动把模型下载到 HuggingFace 缓存目录再运行。

**Q: 评估时为什么有的题显示 no feedback？**
A: 多为裁判 LLM 免费额度耗尽（403 FreeTierOnly）导致。可将 `EVAL_LLM_MODEL` 换成有额度的模型后重跑。

**Q: 运行时报 `[WinError 10060]` 连接超时（卡在加载模型）？**
A: 这是 HuggingFace 库在加载模型时向 `huggingface.co` 发在线探测请求、而该域名不可达导致的。运行一次 `set_env.bat`（或 `.\set_env.ps1`）把 `HF_ENDPOINT` 配成 `https://hf-mirror.com` 即可。

## 致谢

本项目使用了以下开源组件：BAAI 的 BGE 系列模型（向量与重排）、LangGraph / LangChain、阿里云通义千问（DashScope）、FAISS、rank-bm25、jieba、Elasticsearch。
