# 📈 Stock Agent: 具备双轨记忆与多级缓存的高可靠量化分析智能体

Stock Agent 是一个基于 LangChain 核心框架构建的本地化全栈量化分析智能体。

本项目摒弃了市面上常见的“单体 Prompt 脚本”或“简单 API 套壳”模式，旨在展示**构建高可用、高扩展、具备复杂业务落地能力的 AI Agent 的标准工程实践**。系统深度整合了**动态多工具路由**、**双轨记忆架构 (STM + LTM)**、**高性能混合持久化 RAG 引擎**以及**沙箱安全防御**。

## 💡 核心 Agent 设计模式与工程亮点

### 1. 🧠 双轨记忆引擎 (Dual-Track Memory Architecture)
针对长文本交互中大模型“上下文污染”和“Token 成本爆炸”的痛点，设计了长短记忆分离的架构：
* **短期滑动窗口 (STM)**：接管底层的会话历史，通过滑动窗口自动截断冗长的历史问答，仅保留最近 5 轮对话维持短期连贯性。
* **长期 KV 状态机 (LTM)**：赋予 Agent CRUD 级别的记忆能力。通过 Function Calling 将用户的投资偏好、持仓快照抽取为高密度的 Key-Value 数据存入 `user_profile.json`。
* **并发一致性控制**：在工具层引入 `filelock` 文件互斥锁，保障在多线程或多智能体交互下修改长期记忆时的原子性写入，避免数据脏读脏写。

### 2. 🔀 基于系统基准的动态语义路由 (Dynamic Tool Routing)
系统集成了十余种原子级工具，通过严密的 System Prompt 约束与时空坐标注入，实现了精准的意图识别与工具分发：
* **跨国市场解耦**：Agent 会根据用户提问（如苹果、茅台、腾讯），自动解析标的并路由至对应的市场工具链（US-yfinance / A-Share-akshare / HK-yfinance 带格式装甲）。
* **物理时钟注入**：每次调用前通过 `datetime.now()` 动态注入物理时间戳至大模型上下文，彻底消除大模型对“昨天”、“上周”等相对时间概念的幻觉。

### 3. ⚡ 混合架构本地 RAG 引擎 (L1/L2 Cache)
针对本地研报解析的性能痛点，设计了带“热更新”机制的向量数据库持久化层：
* **L1 内存池 (Memory Hash)**：会话内极速命中。
* **L2 硬盘持久化 (FAISS Disk)**：跨进程/跨重启复用，大幅降低频繁 Embedding API 调用成本。
* **MTime 校验刷新**：通过比对目标文件的修改时间戳，实现研报变更后的自动穿透重建，对上层应用完全透明。

### 4. 🛡️ Agent 越权操作防御 (Sandbox Security)
由于赋予了 LLM 本地 I/O 权限，项目在底层构筑了严格的防逃逸逻辑：
* 弃用脆弱的字符串 `startswith`，使用 Python 3.9+ 的 `pathlib.is_relative_to()` 进行绝对层级校验。
* 彻底拦截大模型产生幻觉或遭遇恶意 Prompt 注入时，企图通过 `../` 进行目录遍历攻击（Path Traversal）的风险。

### 5. 💻 终端全息交互UI (Terminal Hacker Experience)
* 融合 `rich` 与 `prompt_toolkit`，重构底层 `CallbackHandler` 实时拦截大模型思考协议流（Thought Process）。
* 原生 Markdown 引擎渲染系统回复，辅以自适应宽度的边界 `Rule`，提供高度专业且极具科技感的工作流反馈。

---

## 🛠️ 技术栈选型

* **大语言模型/编排**: LangChain, OpenAI-Compatible API (DashScope Qwen-Max 驱动)
* **核心运行逻辑**: Python 3.9+, Pydantic, FileLock
* **金融数据支撑**: `yfinance` (美股/港股), `akshare` (A股), `mplfinance`
* **向量检索 (RAG)**: FAISS, RecursiveCharacterTextSplitter, DashScope Embeddings
* **交互界面**: `rich`, `prompt_toolkit`

## 📂 核心系统架构

```text
stock_agent/
├── agent_workspace/    # 🔒 AI 读写沙箱：隔离报告与图片等生成文件
├── knowledge_base/     # 📚 本地知识库：存放待解析的 PDF/Markdown/TXT
├── embeddings/         # 💾 L2 缓存层：FAISS 向量库持久化碎片
├── memory/             # 🧠 状态中枢：存放短期日志与 LTM 状态字典 (user_profile.json)
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


3. 在终端中启动系统：
```bash
python main.py

```