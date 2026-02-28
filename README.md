# 📈 OmniStock Agent: 具备高可用中间件与双轨记忆的量化智能体架构

OmniStock Agent 是一个基于 LangChain 核心框架构建的本地化全栈量化分析智能体。

本项目摒弃了市面上常见的“单体 Prompt 脚本”或“简单 API 套壳”模式，旨在展示**构建高可用、高容错、具备复杂业务落地能力的 AI Agent 的标准工程实践**。系统深度整合了**智能参数适配中间件**、**高可用重试机制**、**双轨记忆架构 (STM + LTM)**、**混合持久化 RAG 引擎**以及**企业级沙箱安全防御**。

## 💡 核心 Agent 设计模式与工程亮点

### 1. ⚙️ 智能参数中间件与适配器模式 (Smart Middleware & Adapter Pattern)
在处理跨全球市场（美股、A股、港股）的底层接口差异时，没有粗暴地依赖大模型 Prompt 进行容易产生幻觉的字符串拼接，而是设计了**职责分离架构**：
* **意图与执行解耦**：大模型仅负责提取用户的核心标的意图（如“腾讯”、“茅台”）。
* **格式化装甲 (`format_universal_ticker`)**：在 Tool 层引入 Python 适配器中间件，利用正则与逻辑自动补齐 `.HK`、`.SS`、`.SZ` 等市场后缀。彻底消除了大模型格式化错误导致的 API 调用失败，显著提升系统鲁棒性。

### 2. 🛡️ 高可用与容错控制 (High Availability & Fault Tolerance)
量化与联网查询系统极易遭遇网络波动。本项目在工具调用层与模型层均实现了企业级容错机制：
* **指数退避重试 (Exponential Backoff)**：引入 `tenacity` 库，为 `get_universal_stock_price` 和 `search_company_ticker` 等外部 I/O 工具挂载拦截器，遭遇 502/超时等网络抖动时自动进行最高 3 次的动态间隔重试。
* **LLM 熔断机制**：在底层的 ChatOpenAI 客户端硬性配置了 45 秒 Request Timeout 与 Max Retries 策略，防止 Agent 在复杂推理时卡死主线程。

### 3. 🧠 双轨记忆引擎 (Dual-Track Memory Architecture)
针对长文本交互中大模型“上下文污染”和“Token 成本爆炸”的痛点，设计了长短记忆分离的架构：
* **短期滑动窗口 (STM)**：接管底层的会话历史，通过滑动窗口自动截断冗长的历史问答，仅保留最近 5 轮对话维持短期连贯性。
* **长期 KV 状态机 (LTM)**：赋予 Agent CRUD 级别的记忆能力。通过 Function Calling 将用户的投资偏好抽取为 Key-Value 数据存入 `user_profile.json`。
* **并发一致性**：引入 `filelock` 文件互斥锁，保障在多线程交互下修改长期记忆时的原子操作，避免脏写。

### 4. ⚡ 混合架构本地 RAG 引擎 (L1/L2 Cache)
针对本地研报解析的性能痛点，设计了带“热更新”机制的向量数据库持久化层：
* **L1 内存池 (Memory Hash)**：会话内极速命中。
* **L2 硬盘持久化 (FAISS Disk)**：跨进程复用，大幅降低频繁 Embedding API 调用成本。
* **MTime 校验刷新**：比对目标文件修改时间戳，实现研报变更后的自动穿透重建，对上层透明。

### 5. 🔒 Agent 越权操作防御 (Sandbox Security)
由于赋予了 LLM 本地 I/O 权限，项目在底层构筑了严格的防逃逸逻辑：
* 弃用脆弱的字符串 `startswith`，使用 Python 3.9+ 的 `pathlib.is_relative_to()` 进行绝对层级校验。
* 彻底拦截大模型产生幻觉或遭遇恶意 Prompt 注入时，企图通过 `../` 进行目录穿越攻击（Path Traversal）的风险。

---

## 🛠️ 技术栈选型

* **大语言模型/编排**: LangChain, OpenAI-Compatible API (DashScope Qwen-Max)
* **核心运行逻辑**: Python 3.9+, Pydantic, FileLock, Tenacity (容错重试)
* **金融数据支撑**: `yfinance`, `akshare`, `mplfinance` (K线图渲染)
* **向量检索 (RAG)**: FAISS, DashScope Embeddings
* **交互体验**: `rich` (Markdown 与 Rule 渲染), `prompt_toolkit`

## 📂 核心系统架构

```text
stock_agent/
├── agent_workspace/    # 🔒 AI 读写沙箱：隔离报告与可视化生成文件
├── knowledge_base/     # 📚 本地知识库：存放待解析的 PDF/Markdown
├── embeddings/         # 💾 L2 缓存层：FAISS 向量库持久化碎片
├── memory/             # 🧠 状态中枢：存放 LTM 状态字典 (user_profile.json)
├── main.py             # 系统核心编排入口
├── .env                # 环境变量配置
└── requirements.txt    # 项目依赖锁定

```

## 🚀 部署与运行

1. 克隆项目并初始化虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```


2. 在根目录配置 `.env` 注入模型密钥：
```env
DASHSCOPE_API_KEY=your_api_key_here

```


3. 启动终端全息交互系统：
```bash
python main.py

```



## 🌟 架构演进思考 (Design Philosophy)

*“不要用 Prompt 解决工程问题。”* 本项目始终贯彻让 LLM 回归 **逻辑推理器 (Reasoning Engine)** 的本质。格式化补全、容错重试、缓存读取、沙箱拦截等工作，均交由底层的确定性 Python 代码（Middlewares & Decorators）处理。这不仅极大降低了 Agent 系统的幻觉率，也是 AI 应用走向企业级落地的必由之路。