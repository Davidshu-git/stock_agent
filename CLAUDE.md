# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 语言与沟通规范

- **全程使用简体中文**：所有回复、分析、终端反馈均须以简体中文输出。
- **语调**：资深工程师风格——专业、精简、直击要害，不废话。
- **终端命令**：提供的所有命令须兼容 macOS（Apple Silicon M1）。

---

## 常用命令

```bash
# 初始化并激活虚拟环境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 启动交互式终端 Agent
./.venv/bin/python main.py

# 启动盘后定时调度器（等待每日 15:30）
./.venv/bin/python daily_job.py

# 立即执行一次调度（测试模式）
./.venv/bin/python daily_job.py --test

# 运行全量测试
pytest tests/ -v
pytest tests/ -v --cov=valuation_engine    # 附带覆盖率

# 运行单个测试类
pytest tests/test_valuation.py::TestCalculatePortfolioValuationHappyPath -v
```

---

## 环境变量（`.env`）

```env
# 必填
DASHSCOPE_API_KEY=          # 主模型推理（Qwen via DashScope）
DASHSCOPE_EMBEDDING_KEY=    # RAG 向量化（text-embedding-v3）

# 可选（邮件推送服务）
SMTP_SERVER=
SMTP_PORT=465
SENDER_EMAIL=
SENDER_PASSWORD=
RECEIVER_EMAIL=
```

两个 DashScope Key 在 `main.py` 启动时立即校验，缺失任一将直接抛出 `ValueError` 终止进程。

---

## 代码规范（强制执行）

### Python 工程化
- **类型提示**：所有核心函数与类必须使用完整的 Type Hints（PEP 484）。
- **Docstring**：核心函数和类必须包含 Google 风格 Docstring，包括 `Args`、`Returns`、`Raises` 三段。
- **异常捕获**：必须捕获具体的 Exception 子类，**严禁使用裸露的 `except:`**。
- **依赖原则**：优先使用标准库；引入新的第三方库前，必须先向用户确认，再修改 `requirements.txt`。

### 本项目专属业务规范
- **网络容错**：所有调用外部 API 的工具函数（yfinance、akshare、DashScope、SMTP）必须配置超时（`timeout`）与 `tenacity` 指数退避重试机制。
- **数据解析防御**：解析外部股票数据或 JSON 时，必须做严格的空值检查（`if df is None or df.empty`）和 `json.JSONDecodeError` 异常捕获，绝不假设数据完整。

---

## 工作流红线（必须遵守）

### 🛑 Git 操作强制确认
**严禁未经允许擅自执行代码提交。** 在准备执行 `git commit`、`git push` 或任何不可逆的系统变更前，**必须暂停操作，在对话中明确询问用户**（如："准备提交，是否确认执行？"），获得明确授权后方可继续。

### 📝 任务完成后的修改总结
每次完成代码修改、Bug 修复或重构后，**必须主动输出简明的修改总结**，包含：
1. 改动了哪些核心文件；
2. 新增或变更了哪些关键逻辑；
3. 下一步建议或潜在风险提示。

### 🐍 虚拟环境强制约束
- **禁止污染系统全局 Python**：执行任何 Python 脚本或安装依赖前，必须确保使用项目虚拟环境。
- **推荐格式（直接路径调用）**：`./.venv/bin/python ...`
- **备选格式**：`source .venv/bin/activate && python ...`
- 如涉及新库，执行测试前须提示用户是否需要先 `pip install`。

---

## 系统架构

### 模块职责

| 文件 | 职责 |
|------|------|
| `main.py` | Agent 编排核心：LangChain 工具注册、STM/LTM 记忆引擎、REPL 主循环、RAG 检索管道 |
| `valuation_engine.py` | 纯 Python 金融引擎：Ticker 格式化中间件、多源查价（yfinance/akshare）、K 线图生成、多币种 CNY 折算与持仓估值 |
| `daily_job.py` | 盘后调度器：多源资讯聚合去重（财联社/新浪/东财）、报告生成、邮件推送、知识库归档 |
| `notifier.py` | SMTP 邮件发送：Markdown→HTML 渲染，tenacity 重试 |
| `tests/test_valuation.py` | `valuation_engine.py` 的单元测试，使用 `pytest` + `unittest.mock` 隔离外部依赖 |

### 关键设计模式

**Ticker 格式化中间件（`format_universal_ticker` in `valuation_engine.py`）**：LLM 只传入用户原始输入（如 `"0700"`），中间件自动补齐 `.HK`、`.SS`、`.SZ` 市场后缀。调用 yfinance 时**绝对不能绕过此函数**。

**双轨记忆架构**：
- **STM**：`FileChatMessageHistory`，10 条消息滑动窗口（`get_session_history()` in `main.py`）。
- **LTM**：`memory/user_profile.json`（KV 状态机），由 `filelock` 保障并发原子写入。交易流水单独写入 `memory/transaction_logs.jsonl`（追加型）。

**持仓记忆格式（硬性约束）**：`user_profile.json` 中持仓条目的 key 为标准股票代码，value 必须严格遵循 `"[中文公司名]，X 股，成本 Y"` 格式。`parse_user_profile_to_positions()` 的正则解析器强依赖关键词 `成本`（不可用同义词替换）和中文名称前置的结构。

**L1/L2 混合 RAG 缓存**：`analyze_local_document` 工具按以下顺序命中缓存：进程内 `FAISS_CACHE` 字典（L1）→ `./embeddings/<file>_vstore/`（L2，附 `meta.json` mtime 校验）→ 均未命中则重建并双向写入。

**沙箱安全防御**：`read_local_file` / `write_local_file` 统一使用 `pathlib.is_relative_to()` 校验路径层级，强制所有 I/O 限制在 `./agent_workspace/` 内，知识库操作限制在 `./knowledge_base/` 内。**严禁改用 `str.startswith()` 替代**，该方法无法防御同级目录绕过。

**估值引擎隔离**：`valuation_engine.py::calculate_portfolio_valuation()` 是系统唯一的财务计算来源。System Prompt 已明确禁止 LLM 自行心算——必须通过 `calculate_exact_portfolio_value` 工具调用。

### 盘后报告数据流

```
daily_job.py::job_routine()
  ├── fetch_global_market_news()     # 并行拉取：财联社 + 新浪 + 东财 → 去重 → top 200
  ├── fetch_global_indices()         # 沪深300 + HSI + HSTECH + NDX（经由 valuation_engine）
  ├── load_user_profile()
  │     └── parse_user_profile_to_positions() + calculate_portfolio_valuation()
  ├── generate_market_report()       # LLM 长文本推理（120s 超时）
  ├── knowledge_base/盘后日报_*.md   # 归档至知识库，沉淀为 RAG 数据源
  └── notifier.send_market_report_email()
```

### 持久化目录

| 目录 | 用途 |
|------|------|
| `agent_workspace/` | AI 读写沙箱，存放生成的报告（`.md`）与 K 线图（`.png`） |
| `knowledge_base/` | RAG 知识库原始文件（PDF/MD/TXT/CSV）+ 归档的盘后日报 |
| `embeddings/` | FAISS 向量库硬盘缓存（每个文档一个子目录） |
| `memory/` | `user_profile.json`（LTM）、`transaction_logs.jsonl`、会话历史 JSON |

### LLM 配置

模型：`qwen3.5-plus`，接入点：`https://coding.dashscope.aliyuncs.com/v1`（OpenAI 兼容协议）。

| 场景 | 超时 | 重试 |
|------|------|------|
| 交互式 Agent（`main.py`） | 45s | 3 次 |
| 盘后报告生成（`daily_job.py`） | 120s | 3 次 |
| 外部 I/O 工具（tenacity） | — | 3 次，等待 2–10s 指数退避 |

### 容器化与生产部署

```bash
# Docker（适合开发调试）
docker compose up -d
# 挂载卷：memory/, agent_workspace/, knowledge_base/, embeddings/, .env

# PM2（生产推荐，用于盘后调度器的守护进程）
pm2 start daily_job.py --name "OmniStock-Agent" --interpreter ./.venv/bin/python
pm2 logs OmniStock-Agent   # 实时日志
pm2 monit                  # 资源监控面板
```
