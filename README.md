# 🤖 Stock Agent: 具备“长效记忆”与“全息可视化”的极客量化智能体

Stock Agent 是一个基于 LangChain 构建的本地极客风 AI 智能体应用。它不仅能够通过互联网检索实时/历史股票数据、绘制专业 K 线图，还具备跨越重启的长效记忆能力与极具赛博朋克风格的终端交互体验。

本项目在底层架构上着重解决了大模型落地的三大工程痛点：**大模型上下文遗忘**、**高频 RAG 检索的性能瓶颈**，以及**本地 I/O 的越权安全问题**。

## ✨ 核心架构与功能亮点

* **🧠 双轨记忆引擎 (Dual-Track Memory)**
    * **短期滑动窗口**：自动截断冗长的历史对话，节省 API Token 并保持响应速度。
    * **长期 KV 状态机**：Agent 会自动提炼你的投资偏好、持仓情况并写入带有 `FileLock` 线程锁保护的长期档案中。即使重启，它依然“懂你”。
* **📈 终端全息可视化 (iTerm2 Inline Image)**
    打破纯文本终端的限制！借助 iTerm2 底层协议，Agent 自动生成的 30 天 K 线图（基于 `mplfinance`）将直接内联显示在你的命令行界面中，图文并茂。
* **🛡️ 企业级安全沙箱 (Path Traversal 防御)**
    本地文件的读写操作被严格限制在 `agent_workspace` 目录下。底层使用 Python 的 `pathlib.is_relative_to()` 进行绝对路径防越权拦截，彻底杜绝恶意 Prompt 注入引发的系统文件泄露。
* **⚡ L1/L2 混合持久化 RAG 引擎**
    * **L1 内存池缓存**：同会话内重复查询零延迟。
    * **L2 硬盘持久化 (`embeddings/`)**：跨进程/重启复用 FAISS 向量索引，极大节省 Embedding API 开销。
* **💻 赛博朋克黑客终端 (Hacker UI)**
    抛弃枯燥的默认日志。使用 `rich` 库构建自适应 UI，实时拦截并高亮打印 Agent 思考流；结合 `prompt_toolkit` 提供支持方向键、历史记录回溯的丝滑输入体验。
* **🛠️ 智能工具链编排**
    集成实时/历史股价查询 (`yfinance`)、K 线图绘制、股票代码盲搜 (`ddgs`) 以及本地知识库解析引擎。

## 📂 目录结构说明

```text
stock_agent/
├── agent_workspace/    # 🔒 AI 操作沙箱：生成的分析报告和图表图片存放在此
├── knowledge_base/     # 📚 本地知识库：放入 PDF、Markdown、TXT 供 Agent 学习
├── embeddings/         # 💾 L2 缓存层：FAISS 向量数据库持久化目录
├── memory/             # 🧠 记忆中枢：存放短期会话与长期特征 (user_profile.json)
├── main.py             # 核心逻辑入口代码
├── .env                # 环境变量配置（需手动创建）
└── .gitignore          # Git 忽略规则

```

## 🚀 快速启动

### 1. 环境准备

确保 Python 版本 >= 3.9，推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

```

安装核心依赖包：

```bash
pip install langchain langchain-openai langchain-community 
pip install yfinance ddgs python-dotenv prompt_toolkit rich filelock
pip install pypdf faiss-cpu tiktoken dashscope pandas matplotlib mplfinance

```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，填入阿里云百炼（或 OpenAI）的 API Key：

```env
DASHSCOPE_API_KEY=sk-你的真实阿里云密钥

```

### 3. 运行智能体

一切准备就绪，使用 **iTerm2** 终端（为了体验图片内联功能）启动：

```bash
python main.py

```

## 🛠️ 交互示例

你可以直接在终端向它下达复合型指令：

> **“帮我查一下特斯拉在 2024年1月15日 的数据，并画出它最近30天的走势图，结合我的持仓偏好，给我写一份研报并存到本地沙箱里。”**

Agent 将自动规划路线：
读取长期记忆 -> 查历史股价 -> 绘制并保存图表 -> 渲染 iTerm2 全息图表 -> 输出富文本分析报告。

## ⚠️ 注意事项

* **视觉渲染**：终端图片直出功能高度依赖 **iTerm2** 终端模拟器。如果在其他终端（如 Windows Terminal, VSCode 终端）中运行，仅能看到图片保存路径，不会内联渲染。
* 本项目默认基于阿里云 `qwen3.5-plus` / `qwen-max` 驱动。