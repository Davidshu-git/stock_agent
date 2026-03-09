# 📈 OmniStock Agent: 具备高可用中间件与双轨记忆的量化智能体架构

OmniStock Agent 是一个基于 LangChain 核心框架构建的本地化全栈量化分析智能体。

本项目摒弃了市面上常见的“单体 Prompt 脚本”或“简单 API 套壳”模式，旨在展示**构建高可用、高容错、具备复杂业务落地能力的 AI Agent 的标准工程实践**。系统深度整合了**智能参数适配中间件**、**高可用重试机制**、**双轨记忆架构 (STM + LTM)**、**混合持久化 RAG 引擎**、**企业级沙箱安全防御**以及**可视化图表生成**等功能。

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

### 6. 📊 可视化图表生成 (Chart Visualization)
* **K线图与趋势图生成**：`draw_universal_stock_chart` 工具可生成30天的K线图和交易量可视化图表，支持多市场股票。
* **图表嵌入机制**：生成的PNG图表可在Markdown报告中直接引用显示，提升分析报告的可读性。

### 7. 🎨 Hacker-Style 交互体验 (Enhanced Terminal UI)
* **自定义回调处理器**：`HackerMatrixCallback` 类提供了独特的赛博朋克风格输出，拦截 Agent 的思考过程和工具调用过程。
* **富文本渲染**：使用 `rich` 库实现 Markdown 格式化输出、自适应分隔线和终端美化。
* **高级命令行交互**：`prompt_toolkit` 提供历史记录、光标移动和输入增强功能。

### 8. 📧 旁路邮件推送调度系统 (Daily Job Scheduler)
* **定时任务调度**：基于 `schedule` 库实现常驻后台的盘后调度器，每日 15:30 自动执行。
* **智能热点分析**：并行拉取财联社/新浪/东财 3 个数据源，多源聚合去重后采用 200 条核心资讯，结合用户持仓记忆生成个性化报告。
* **邮件推送服务**：支持 Markdown 转 HTML 渲染，通过 SMTP 协议发送盘后报告到指定邮箱。
* **优雅降级机制**：用户记忆文件缺失或 API 异常时自动降级，不影响调度器稳定运行。

### 9. 🧮 纯 Python 多币种估值引擎 (Valuation Engine)
* **前后端解耦架构**：通过提取 `valuation_engine.py` 独立模块，接管所有跨市场查价与汇率折算逻辑（基于 `akshare` 和 `yfinance` 实现 USD/HKD 到 CNY 的统一折算）。
* **剥夺大模型数学计算权**：由 CPU 的 ALU（算术逻辑单元）提供 100% 精确的单票盈亏计算和按市值降序排版，彻底斩断了 LLM 在财务计算上的幻觉。
* **DRY 原则实践**：所有估值逻辑集中管理，避免代码重复，支持多币种持仓的统一核算与可视化报表生成。

### 10. 🔄 主被动融合的双重调度机制
* **被动静默推送**：每日 15:30 的 `schedule` 常驻后台调度，自动执行盘后分析并推送研报。
* **主动终端唤醒**：在 `main.py` 终端开放 `trigger_daily_report` 权限，允许用户随时通过对话拉起后台链路，实现毫秒级的研报主动下发。
* **双模无缝切换**：同一套业务逻辑同时支持定时任务与即时调用，满足不同场景下的研报生成需求。

### 11. 🧪 自动化单元测试防线 (Automated Testing)
* **pytest 测试框架**: 基于 `pytest` 和 `pytest-mock` 构建完整的单元测试套件。
* **核心估值引擎测试**: 覆盖 `calculate_portfolio_valuation` 函数的正常场景、边界情况和异常处理。
* **Mock 隔离外部依赖**: 使用 `unittest.mock` 拦截网络请求，确保测试的确定性和快速执行。

---

## 🛠️ 技术栈选型

* **大语言模型/编排**: LangChain, OpenAI-Compatible API (DashScope Qwen3.5-Plus)
* **核心运行逻辑**: Python 3.9+, Pydantic, FileLock (文件锁), Tenacity (容错重试)
* **金融数据支撑**: `yfinance` (全球股票数据), `mplfinance` (K 线图渲染), `akshare` (A 股数据源)
* **向量检索 (RAG)**: FAISS, DashScope Embeddings, PyPDFLoader, RecursiveCharacterTextSplitter
* **交互体验**: `rich` (Markdown 与 Rule 渲染), `prompt_toolkit` (高级终端交互)
* **网络搜索**: `ddgs` (DuckDuckGo Search)
* **文档处理**: `pypdf`, `langchain-community` (文档加载器)
* **系统工具**: `python-dotenv` (环境变量), `filelock` (原子锁)
* **邮件推送**: `markdown` (Markdown 转 HTML), `schedule` (定时任务调度)

## 📂 核心系统架构

```text
stock_agent/
├── agent_workspace/    # 🔒 AI 读写沙箱：隔离报告与可视化生成文件
├── knowledge_base/     # 📚 本地知识库：存放待解析的 PDF/Markdown
├── embeddings/         # 💾 L2 缓存层：FAISS 向量库持久化碎片
├── memory/             # 🧠 状态中枢：存放 LTM 状态字典 (user_profile.json)
├── main.py             # 系统核心编排入口
├── daily_job.py        # 📧 盘后调度主程序（每日 15:30 执行）
├── notifier.py         # 📧 邮件推送模块（SMTP + Markdown 渲染）
├── valuation_engine.py # 🧮 财务核心：多币种转换、精确盈亏计算与 K 线渲染
├── .env                # 环境变量配置
├── requirements.txt    # 项目依赖锁定
└── README.md           # 系统架构与部署文档

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
# 主模型 API Key（用于聊天推理）
DASHSCOPE_API_KEY=your_api_key_here

# Embedding 模型 API Key（用于向量检索 RAG）
DASHSCOPE_EMBEDDING_KEY=your_embedding_api_key_here

# 邮件推送服务配置（可选，用于盘后报告推送）
SMTP_SERVER=smtp.example.com
SMTP_PORT=465
SENDER_EMAIL=your_sender_email@example.com
SENDER_PASSWORD=your_email_password
RECEIVER_EMAIL=your_receiver_email@example.com

```

3. 启动终端全息交互系统：
```bash
python main.py

```

4. （可选）启动盘后调度器：
```bash
# 生产模式：每日 15:30 自动执行盘后分析并发送邮件
python daily_job.py

# 测试模式：立即执行一次，验证功能是否正常
python daily_job.py --test
```

5. 运行所有单元测试：
```bash
# 运行全部测试用例（详细输出）
pytest tests/ -v

# 运行测试并显示覆盖率
pytest tests/ -v --cov=valuation_engine

```

## 🛠️ 核心功能工具集

### 股票查询类工具
* `get_universal_stock_price`: 全球股票查价引擎（支持美股、A 股、港股）
* `draw_universal_stock_chart`: 全球股票 30 天走势绘图引擎（K 线图和交易量）

### 搜索与检索类工具
* `search_company_ticker`: 联网搜索公司股票代码
* `list_kb_files`: 查看知识库中所有支持的文件
* `analyze_local_document`: 分析知识库中文档并回答问题

### 文件操作类工具
* `read_local_file`: 读取沙箱内本地文件
* `write_local_file`: 写入文件到沙箱（强制交付通道）

### 记忆系统工具
* `update_user_memory`: 更新用户长期记忆（KV 状态机）
* `append_transaction_log`: 追加交易日志记录

### 邮件推送与调度工具
* `notifier.py`: 邮件推送模块，支持 Markdown 转 HTML 渲染
* `daily_job.py`: 盘后调度主程序，每日 15:30 自动执行热点分析与邮件推送
* `trigger_daily_report`: 终端主动唤醒盘后调度链路的触发器

### 财务核算工具
* `calculate_exact_portfolio_value`: 个人持仓精确总市值与盈亏核算器（支持多币种自动折算为 CNY）

## 🌟 架构演进思考 (Design Philosophy)

*“不要用 Prompt 解决工程问题。”* 本项目始终贯彻让 LLM 回归 **逻辑推理器 (Reasoning Engine)** 的本质。格式化补全、容错重试、缓存读取、沙箱拦截等工作，均交由底层的确定性 Python 代码（Middlewares & Decorators）处理。这不仅极大降低了 Agent 系统的幻觉率，也是 AI 应用走向企业级落地的必由之路。

## 📋 环境配置与安全注意事项

### 必需环境变量
- `DASHSCOPE_API_KEY`: 用于主模型推理的 API 密钥
- `DASHSCOPE_EMBEDDING_KEY`: 用于向量检索的 Embedding API 密钥

### 可选环境变量（邮件推送服务）
- `SMTP_SERVER`: SMTP 服务器地址（如 `smtp.qq.com`）
- `SMTP_PORT`: SMTP 端口（SSL 推荐 465）
- `SENDER_EMAIL`: 发件人邮箱地址
- `SENDER_PASSWORD`: 发件人邮箱密码或授权码
- `RECEIVER_EMAIL`: 收件人邮箱地址

### 安全特性
- **沙箱隔离**: 所有文件读写操作限制在 `agent_workspace/` 目录内
- **文件类型白名单**: 仅允许读取 `.pdf`, `.md`, `.txt`, `.csv` 格式的文件
- **路径穿越防护**: 使用 `pathlib.is_relative_to()` 防止目录遍历攻击
- **并发安全**: 使用 `filelock` 确保记忆文件的原子写入操作

## 🚀 生产环境部署 (基于 PM2 的高可用守护)

为了确保每日盘后（15:30）研报能够极其稳定地自动生成并推送到邮箱，本项目推荐使用 [PM2](https://pm2.keymetrics.io/) 作为进程管家，实现崩溃自启与后台常驻。

### 1. 环境准备
确保你的生产环境（Linux / WSL）已安装 Node.js 与 PM2：
```bash
# Ubuntu / Debian / WSL
sudo apt update
sudo apt install nodejs npm -y
sudo npm install pm2 -g

```

### 2. 一键挂载守护进程

在项目根目录下执行以下命令。**注意：必须使用 `--interpreter` 强制指定虚拟环境中的 Python 解释器**，否则会导致依赖包（如 akshare）丢失。

```bash
# 启动并命名进程为 OmniStock-Agent
pm2 start daily_job.py --name "OmniStock-Agent" --interpreter ./.venv/bin/python

```

### 3. 系统可观测性 (日志与监控)

本项目内置了基于 `rich` 的全链路彩色溯源日志。在 PM2 挂载后，你可以随时通过以下命令切入监控台：

```bash
# 实时查看彩色抓取日志与投递状态
pm2 logs OmniStock-Agent

# 调出极客监控面板（CPU/内存占用、运行时间）
pm2 monit

```

### 4. 💡 进阶：Windows / WSL 无人值守保活配置

如果你使用 Windows 物理机配合 WSL 作为 24 小时服务器，**必须防范 Windows 自动休眠与重启导致 WSL 进程被物理冻结**。

**步骤一：关闭休眠**
进入 Windows 设置 -> 电源和睡眠，将“使计算机进入睡眠状态”改为 **从不**。

**步骤二：配置开机静默唤醒**

1. 在 Windows 桌面新建 `start_wsl_agent.vbs` 文件，写入以下代码：
```vbscript
Set ws = CreateObject("Wscript.Shell")
ws.run "wsl -e pm2 resurrect", 0

```


2. 按下 `Win + R`，输入 `shell:startup` 进入启动文件夹。
3. 将该 `.vbs` 文件拖入其中。此后，即使 Windows 半夜强制更新重启，系统也能在后台连黑框都不闪地无感拉起 WSL 和 PM2 守护进程。