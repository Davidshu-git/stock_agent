# 🤖 Stock Agent: 极客风全栈量化分析与本地知识库智能体

Stock Agent 是一个基于 LangChain 构建的本地智能体应用。它不仅能通过互联网检索实时股票数据，还能极其安全、高效地读取本地知识库（PDF/Markdown 等），并自动生成排版精美的研报保存到本地沙箱中。

本项目在底层架构上深度优化了 **RAG 检索性能**（L1/L2 混合缓存）与 **本地 I/O 安全性**（防沙箱逃逸），为极客和量化投资者提供极其丝滑的终端交互体验。


## ✨ 核心架构与功能亮点

* **🛡️ 企业级安全沙箱 (Path Traversal 防御)**
    本地文件的读写操作（`read_local_file` / `write_local_file`）被严格限制在 `agent_workspace` 目录下。底层使用 Python 3.9+ 的 `pathlib.is_relative_to()` 进行绝对路径防越权拦截，彻底杜绝恶意 Prompt 注入导致的系统文件泄露。
* **⚡ L1/L2 混合持久化 RAG 引擎**
    针对本地文档检索进行了极致优化：
    * **L1 内存池缓存**：同会话内重复查询零延迟。
    * **L2 硬盘持久化 (`embeddings/`)**：跨进程/重启复用 FAISS 向量索引，极大节省大模型 Embedding API 开销。
    * **热更新机制**：自动校验文件修改时间戳 (`mtime`)，文档修改后自动重建对应索引，无需重启服务。
* **💻 赛博朋克终端交互 (Hacker UI)**
    抛弃枯燥的默认日志。使用 `rich` 库重写 Callback 拦截器，实时高亮打印 Agent 思考流与工具调度指令；结合 `prompt_toolkit` 提供支持方向键、历史记录（上翻/下翻）的丝滑输入体验。
* **🧠 智能工具链编排**
    集成 6 大核心插件，支持雅虎财经实时报价 (`yfinance`)、股票代码盲搜 (`ddgs`) 以及全格式本地文件管理。

## 📂 目录结构说明

程序首次运行后，会自动在根目录创建以下关键文件夹（均已加入 `.gitignore` 防止敏感数据外泄）：

```text
stock_agent/
├── agent_workspace/    # 🔒 AI 操作沙箱：Agent 自动生成的分析报告和文件会存放在这里
├── knowledge_base/     # 📚 本地知识库：把你的 PDF、Markdown、TXT 研报扔进这里，供 Agent 学习
├── embeddings/         # 💾 L2 缓存层：自动生成的 FAISS 向量数据库持久化目录
├── main.py             # 核心逻辑入口代码
├── .env                # 环境变量配置（需手动创建）
└── .gitignore          # Git 忽略规则

```

## 🚀 快速启动

### 1. 环境准备

确保你的 Python 版本 >= 3.9，推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate # Windows

```

安装核心依赖包：

```bash
pip install langchain langchain-openai langchain-community 
pip install yfinance ddgs python-dotenv prompt_toolkit rich
pip install pypdf faiss-cpu tiktoken dashscope

```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，并填入你的阿里云百炼 API Key（兼容 OpenAI 接口）：

```env
DASHSCOPE_API_KEY=sk-你的真实阿里云密钥

```

### 3. 运行智能体

一切准备就绪，启动极客终端：

```bash
python main.py

```

## 🛠️ 交互示例

你可以直接在终端向它下达复合型指令：

> **“对比一下微软和苹果昨天的表现。然后看看知识库里有没有相关的研报，综合一下核心数据，最后帮我把分析结果写成一份 markdown 报告存到本地。”**

Agent 将自动规划路线：

1. 搜索代码 -> 2. 查股价 -> 3. 扫描目录 -> 4. 建立索引并检索文档 -> 5. 撰写并保存 Markdown 报告。

## ⚠️ 注意事项

* 本项目默认使用 `qwen-max` 模型驱动，因其具备极强的逻辑规划和 Function Calling 能力。你也可以在代码中无缝切换为 OpenAI 的 GPT-4o 或其他兼容大模型。
* FAISS 的本地加载开启了 `allow_dangerous_deserialization=True`，因向量文件均由本地生成，属安全行为。请勿随意将不明来源的 `.pkl` 文件放入 `embeddings` 目录。