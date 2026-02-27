# Stock Agent: 具备多级缓存与状态机记忆的智能体架构

Stock Agent 是一个基于 LangChain 构建的本地化 AI 智能体应用。本项目并非简单的 LLM API 套壳，而是在底层架构上重点攻克了 **大模型上下文遗忘**、**高频 RAG 检索性能瓶颈** 以及 **本地文件 I/O 安全** 等真实落地场景中的工程痛点。

本项目旨在探索如何以极其轻量级的方式（零重量级数据库依赖），在本地构建一个高可用、状态一致且安全的专属量化分析智能体。

## 💡 核心工程实践与技术亮点

### 1. 混合架构 RAG 引擎 (L1/L2 Cache)
针对本地文档检索（PDF/Markdown 等）进行了存储分层级的极致优化，极大地降低了 Embedding API 开销与响应延迟：
* **L1 内存池 (Memory Dict)**：实现同会话内热点文档重复查询的“零延迟”命中。
* **L2 硬盘持久化 (FAISS Disk Serialization)**：跨进程/跨重启复用向量索引。
* **热更新机制**：通过比对目标文件的修改时间戳 (`mtime`)，实现文档变更后的平滑重建与缓存穿透写入。

### 2. 双轨记忆系统与一致性保障 (Dual-Track Memory)
摈弃了粗暴的全量历史上下文拼接，引入了“短/长期分离”的记忆流：
* **短期滑动窗口**：自动按阈值截断历史对话，控制 Token 爆炸并保证推理速度。
* **长期 KV 状态机**：通过 Function Calling 提取用户的投资偏好与核心持仓，更新至本地状态字典。
* **并发控制**：引入 `FileLock` 文件锁机制，保障在异步或多智能体场景下写长期记忆时的文件一致性（原子操作），避免脏读脏写。

### 3. 企业级安全沙箱 (Path Traversal 防御)
在将本地文件系统的读写权限 (`read_local_file` / `write_local_file`) 赋予 LLM 时，实施了严格的安全防御：
* 弃用脆弱的字符串 `startswith` 匹配，底层采用 Python 3.9+ 的 `pathlib.is_relative_to()` 进行层级校验。
* 彻底杜绝 LLM 产生幻觉或遭受恶意 Prompt 注入时，通过 `../../` 目录穿越漏洞导致的系统级文件外泄。

### 4. 终端原生视觉渲染 (iTerm2 Inline Engine)
打通了终端底层协议与数据可视化库的链路：
* 拦截 `mplfinance` 生成的本地金融图表，将其编码为 Base64 流。
* 配合自研正则切割模块提取大模型输出的 Markdown 语法，实现富文本与 K 线图在 iTerm2 终端的原生内联渲染，打造极佳的极客级 CLI 体验。

## 🛠️ 技术栈选型

* **LLM Orchestration**: LangChain, LangChain-OpenAI
* **Core Logic & Tools**: Python 3.9+, Pydantic
* **Vector Store & Embeddings**: FAISS, DashScope Embeddings
* **Financial Data & Viz**: `yfinance`, `pandas`, `mplfinance`
* **UI & Interaction**: `rich`, `prompt_toolkit`
* **State Management**: 本地 JSON + `filelock`

## 📂 系统架构目录

```text
stock_agent/
├── agent_workspace/    # 🔒 AI 读写沙箱：隔离报告与图片等生成文件
├── knowledge_base/     # 📚 本地知识库：存放待向量化的源文档
├── embeddings/         # 💾 L2 缓存层：FAISS 向量库持久化切片目录
├── memory/             # 🧠 状态中枢：存放短期日志与持久化特征 (user_profile.json)
├── main.py             # 核心系统编排入口
├── .env                # 环境变量 (API Keys 等)
└── requirements.txt    # 锁定版本依赖

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