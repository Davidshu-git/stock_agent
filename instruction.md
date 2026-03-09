# OmniStock Agent 项目简述

## 项目概述

OmniStock Agent 是一个基于 LangChain 框架构建的**本地化全栈量化分析智能体**，专为全球股票市场（美股/A 股/港股）提供智能查询、可视化分析和深度研报生成服务。

## 核心功能

| 功能模块 | 工具名称 | 描述 |
|---------|---------|------|
| 股票查价 | `get_universal_stock_price` | 支持指定日期查询全球股价 |
| K 线绘图 | `draw_universal_stock_chart` | 生成 30 天走势可视化图表 |
| 代码搜索 | `search_company_ticker` | 通过公司名搜索股票代码 |
| 文件读写 | `read_local_file` / `write_local_file` | 沙箱内的安全文件操作 |
| 文档分析 | `analyze_local_document` | RAG 检索本地知识库 |
| 记忆管理 | `update_user_memory` / `append_transaction_log` | 长期记忆与交易日志 |
| 市值核算 | `calculate_exact_portfolio_value` | 多币种持仓精确总市值与盈亏核算（自动折算 CNY） |
| 主动触发 | `trigger_daily_report` | 终端主动唤醒盘后调度链路，毫秒级研报下发 |

## 架构亮点

### 1. 智能参数适配器
`format_universal_ticker()` 中间件自动推断市场并格式化股票代码（如 `600519` → `600519.SS`），消除大模型幻觉导致的 API 调用失败。

### 2. 高可用容错机制
- **Tenacity 指数退避重试**: 外部 I/O 工具遭遇网络波动时自动重试 3 次
- **LLM 熔断保护**: 45 秒超时阈值 + 自动重试策略

### 3. 双轨记忆架构
- **短期记忆 (STM)**: 滑动窗口保留最近 5 轮对话
- **长期记忆 (LTM)**: KV 状态机存储用户偏好与持仓快照
- **文件锁保护**: `filelock` 确保并发写入的原子性

### 4. 混合持久化 RAG 引擎
- **L1 内存缓存**: 会话内极速复用向量索引
- **L2 硬盘缓存**: FAISS 持久化存储跨进程共享
- **MTime 校验**: 文件变更自动穿透重建

### 5. 沙箱安全防御
使用 `pathlib.is_relative_to()` 严格校验路径层级，彻底阻断 `../` 目录穿越攻击。

### 6. 纯 Python 多币种估值引擎
`valuation_engine.py` 独立模块接管所有跨市场查价与汇率折算，由 CPU ALU 提供 100% 精确的单票盈亏计算，彻底斩断 LLM 财务计算幻觉。

### 7. 主被动融合双重调度
支持每日 15:30 被动静默推送 + 终端 `trigger_daily_report` 主动唤醒，同一业务逻辑满足定时与即时双场景需求。

## 目录结构

```
stock_agent/
├── agent_workspace/    # AI 读写沙箱
├── knowledge_base/     # 本地知识库 (PDF/Markdown)
├── embeddings/         # FAISS 向量库缓存
├── memory/             # 长期记忆存储
├── main.py             # 核心入口
├── valuation_engine.py # 财务核心：多币种转换、精确盈亏计算
├── requirements.txt    # 依赖清单
└── .env                # 环境变量配置
```

## 技术栈

- **LLM 编排**: LangChain + DashScope (Qwen-Max)
- **数据源**: yfinance (全球股票数据)
- **可视化**: mplfinance (K 线图渲染)
- **向量检索**: FAISS + DashScope Embeddings
- **终端交互**: rich + prompt_toolkit

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥
echo "DASHSCOPE_API_KEY=your_key" > .env

# 3. 启动 Agent
python main.py
```

## 设计哲学

> "不要用 Prompt 解决工程问题。"

本项目将格式化补全、容错重试、缓存读取、沙箱拦截等确定性工作交由底层 Python 代码处理，让 LLM 回归逻辑推理器的本质，显著降低系统幻觉率，实现企业级落地标准。
