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

# 安装 Playwright 无头浏览器内核（首次安装后需执行一次）
./.venv/bin/playwright install chromium

# 启动交互式终端 Agent
./.venv/bin/python main.py

# 启动 Telegram Bot（移动端交互入口）
./.venv/bin/python tg_main.py

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

# Telegram Bot（启动 tg_main.py 必填）
TG_BOT_TOKEN=               # BotFather 获取的 Bot Token
ALLOWED_TG_USERS=           # 白名单用户 ID，逗号分隔（如 123456,789012）

# 可选（邮件推送服务）
SMTP_SERVER=
SMTP_PORT=465
SENDER_EMAIL=
SENDER_PASSWORD=
RECEIVER_EMAIL=
```

- `DASHSCOPE_API_KEY` / `DASHSCOPE_EMBEDDING_KEY`：在 `main.py` 启动时立即校验，缺失任一直接 `ValueError` 终止。
- `TG_BOT_TOKEN`：在 `tg_main.py` 启动时校验，缺失直接终止。
- `ALLOWED_TG_USERS`：Bot 的单用户白名单，所有未授权请求由 `@authorized` 装饰器统一拦截返回 403。

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
| `tg_main.py` | Telegram Bot 移动端交互层：白名单认证、Inline Keyboard 操控盘、异步 Agent 回调、Playwright 表格渲染、盯盘预警轮询、跨进程广播接收 |
| `spawn_job.py` | 研报独立进程启动器：由 `trigger_daily_report` 工具以 `subprocess.Popen` 启动，写入任务状态文件与日志 |
| `valuation_engine.py` | 纯 Python 金融引擎：Ticker 格式化中间件、多源查价（yfinance/akshare）、K 线图生成、多币种 CNY 折算与持仓估值 |
| `daily_job.py` | 盘后调度器：多源资讯聚合去重（财联社/新浪/东财）、报告生成、邮件推送、知识库归档、Telegram 广播推送 |
| `notifier.py` | SMTP 邮件发送：Markdown→HTML 渲染，tenacity 重试 |
| `tests/test_valuation.py` | `valuation_engine.py` 的单元测试，使用 `pytest` + `unittest.mock` 隔离外部依赖 |
| `tests/test_tg_renderer.py` | Telegram 渲染引擎单元测试：HTML 转换、表格识别、消息拆分逻辑 |

### 关键设计模式

**Ticker 格式化中间件（`format_universal_ticker` in `valuation_engine.py`）**：LLM 只传入用户原始输入（如 `"0700"`），中间件自动补齐 `.HK`、`.SS`、`.SZ` 市场后缀。调用 yfinance 时**绝对不能绕过此函数**。

**双轨记忆架构**：
- **STM**：`FileChatMessageHistory`，10 条消息滑动窗口（`get_session_history()` in `main.py`）。
- **LTM**：`memory/user_profile.json`（KV 状态机），由 `filelock` 保障并发原子写入。交易流水单独写入 `memory/transaction_logs.jsonl`（追加型）。

**持仓记忆格式（硬性约束）**：`user_profile.json` 中持仓条目的 key 为标准股票代码，value 必须严格遵循 `"[中文公司名]，X 股，成本 Y"` 格式。`parse_user_profile_to_positions()` 的正则解析器强依赖关键词 `成本`（不可用同义词替换）和中文名称前置的结构。

**L1/L2 混合 RAG 缓存**：`analyze_local_document` 工具按以下顺序命中缓存：进程内 `FAISS_CACHE` 字典（L1）→ `./embeddings/<file>_vstore/`（L2，附 `meta.json` mtime 校验）→ 均未命中则重建并双向写入。

**沙箱安全防御**：`read_local_file` / `write_local_file` 统一使用 `pathlib.is_relative_to()` 校验路径层级，强制所有 I/O 限制在 `./agent_workspace/` 内，知识库操作限制在 `./knowledge_base/` 内。**严禁改用 `str.startswith()` 替代**，该方法无法防御同级目录绕过。

**估值引擎隔离**：`valuation_engine.py::calculate_portfolio_valuation()` 是系统唯一的财务计算来源。System Prompt 已明确禁止 LLM 自行心算——必须通过 `calculate_exact_portfolio_value` 工具调用。

**研报触发器独立进程架构**：`trigger_daily_report` 工具不再阻塞主线程，改为以 `subprocess.Popen` 启动 `spawn_job.py` 作为独立子进程；任务状态写入 `jobs/status/{job_id}.json`，日志写入 `jobs/logs/{job_id}.log`。可通过 `query_job_status(job_id)` 工具异步查询进度，状态枚举为 `pending → running → completed/failed`。

**价格预警系统**：`create_price_alert(ticker, operator, target_price)` 将预警规则写入 `memory/alerts.json`（按用户 chat_id 分组）。`tg_main.py` 中的 `price_watcher_routine` 每 5 分钟轮询一次，命中条件后通过 Telegram 推送通知并自动删除该条规则（一次性触发）。

**Telegram Bot 渲染管道**（`tg_main.py`）：
- LLM 输出先经 `translate_to_telegram_html()` 转为 Telegram 兼容的 HTML（支持加粗、代码块、4 级标题降级兜底）。
- 若检测到 Markdown 表格，则调用 `render_markdown_table_to_image()` 启动 Playwright 无头浏览器渲染为高分辨率 PNG（3x DSF 超采样），以图片形式发送。
- 超长消息由 `send_with_caption_split()` 自动分段，防止超出 Telegram 单条消息字数上限。
- Agent 推理过程中通过 `AsyncTelegramCallbackHandler` 实时上报工具调用状态，配合 `keep_typing_action()` 心跳维持"正在输入"状态，消除用户等待焦虑。

### 盘后报告数据流

```
daily_job.py::job_routine()
  ├── fetch_global_market_news()        # 并行拉取：财联社 + 新浪 + 东财 → 去重 → top 200
  ├── fetch_global_indices()            # 沪深300 + HSI + HSTECH + NDX（经由 valuation_engine）
  ├── load_user_profile()
  │     └── parse_user_profile_to_positions() + calculate_portfolio_valuation()
  ├── generate_market_report()          # LLM 长文本推理（120s 超时）
  ├── knowledge_base/盘后日报_*.md      # 归档至知识库，沉淀为 RAG 数据源
  ├── notifier.send_market_report_email()
  └── broadcast_to_telegram()          # 跨进程广播至 tg_main.py，推送至 Telegram
```

### 研报触发数据流（独立进程）

```
用户（终端/Telegram Bot）
  └── trigger_daily_report 工具
        └── subprocess.Popen("spawn_job.py --job-id job_xxx")
              ├── 写入 jobs/status/job_xxx.json  (status: running)
              ├── 写入 jobs/logs/job_xxx.log
              ├── job_routine()
              └── 写入 jobs/status/job_xxx.json  (status: completed/failed)
```

### Telegram Bot 命令路由

| 命令 | 处理函数 | 说明 |
|------|---------|------|
| `/start` | `start_command` | 发送 Inline Keyboard 操控面板 |
| `/portfolio` | `portfolio_command` | 触发持仓估值计算，返回 Markdown 表格图片 |
| `/report` | `report_command` | 投递研报生成任务至独立进程 |
| `/status` | `status_command` | 查询最近一次研报任务状态 |
| `/kb` | `kb_command` | 列出知识库文件 |
| `/alert` | `alert_command` | 引导用户设定盯盘价格预警 |
| 普通文本 | `handle_message` | 路由至 Agent 推理引擎 |
| 文件上传 | `handle_document` | 自动存入知识库并触发 RAG 总结 |

### 持久化目录

| 目录 | 用途 |
|------|------|
| `agent_workspace/` | AI 读写沙箱，存放生成的报告（`.md`）与 K 线图（`.png`） |
| `knowledge_base/` | RAG 知识库原始文件（PDF/MD/TXT/CSV）+ 归档的盘后日报 |
| `embeddings/` | FAISS 向量库硬盘缓存（每个文档一个子目录） |
| `memory/` | `user_profile.json`（LTM）、`transaction_logs.jsonl`、`alerts.json`（价格预警规则）、会话历史 JSON |
| `jobs/status/` | 研报任务状态文件（`{job_id}.json`，独立进程写入） |
| `jobs/logs/` | 研报任务执行日志（`{job_id}.log`，独立进程写入） |

### LLM 配置

模型：`qwen3.5-plus`，接入点：`https://coding.dashscope.aliyuncs.com/v1`（OpenAI 兼容协议）。

| 场景 | 超时 | 重试 |
|------|------|------|
| 交互式 Agent（`main.py`） | 45s | 3 次 |
| 盘后报告生成（`daily_job.py`） | 120s | 3 次 |
| 外部 I/O 工具（tenacity） | — | 3 次，等待 2–10s 指数退避 |

### 核心依赖一览（新增部分）

| 库 | 用途 |
|----|------|
| `python-telegram-bot~=22.6` | Telegram Bot API，`tg_main.py` 的底层驱动 |
| `playwright~=1.58.0` | 无头浏览器，渲染 Markdown 表格为高分辨率 PNG |
| `langgraph~=0.2.0` | 多智能体研报辩论引擎（LangGraph 状态机） |

首次使用 Playwright 需额外执行：`./.venv/bin/playwright install chromium`

### 容器化与生产部署

```bash
# Docker（现为双服务架构）
docker compose up -d
# omnistock-daily-report：盘后调度器
# tg-bot：Telegram Bot（depends_on daily-report）
# 共享挂载卷：memory/, agent_workspace/, knowledge_base/, embeddings/, jobs/

# PM2（生产推荐，分别守护两个进程）
pm2 start daily_job.py --name "OmniStock-Scheduler" --interpreter ./.venv/bin/python
pm2 start tg_main.py   --name "OmniStock-TGBot"     --interpreter ./.venv/bin/python
pm2 logs OmniStock-TGBot   # 实时日志
pm2 monit                  # 资源监控面板
```
