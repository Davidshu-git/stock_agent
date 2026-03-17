import os
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from filelock import FileLock
from langchain_core.tools import tool
from valuation_engine import (
    fetch_stock_price_raw,
    fetch_etf_price_raw,
    generate_kline_chart,
    calculate_portfolio_valuation,
    parse_user_profile_to_positions,
    format_portfolio_report,
)
from daily_job import job_routine
# 使用openai 兼容千问
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS
# 新增：用于管理记忆的模块
from langchain_core.runnables.history import RunnableWithMessageHistory
# 新增：用于长效记忆持久化的模块
from langchain_community.chat_message_histories import FileChatMessageHistory
# 新增：用于RAG的模块
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 新增这个专门针对阿里云的引用
from langchain_community.embeddings import DashScopeEmbeddings
# 新增：引入高级终端交互库
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
# 优化显示效果
from rich.console import Console
from rich.markdown import Markdown   # 🌟 新增：Markdown 渲染引擎
from rich.rule import Rule           # 🌟 新增：自适应分隔线组件
from langchain.callbacks.base import BaseCallbackHandler
#添加超时处理逻辑
from tenacity import retry, stop_after_attempt, wait_exponential
# 🌟 LRU 缓存装饰器，用于 L1 内存池管理
from functools import lru_cache

# 初始化富文本控制台
console = Console()

# 在程序启动的最开始，调用 load_dotenv()
# 它会自动在当前目录寻找 .env 文件，并把里面的值载入到系统环境变量中
load_dotenv()

# 安全地获取 Key
# 使用 os.getenv() 获取，如果 .env 里没配，它会返回 None 而不是直接崩溃
dashscope_key = os.getenv("DASHSCOPE_API_KEY")
embedding_key = os.getenv("DASHSCOPE_EMBEDDING_KEY")

# 在程序刚启动时就检查关键依赖，如果没配 Key，立刻阻断并报错，而不是等跑了一半才死掉
if not dashscope_key:
    raise ValueError("❌ 致命错误：未在 .env 文件或环境变量中找到 DASHSCOPE_API_KEY！请检查配置。")

if not embedding_key:
    raise ValueError("❌ 致命错误：未在 .env 文件或环境变量中找到 DASHSCOPE_EMBEDDING_KEY！请检查配置。")

# 安全配置：定义 Agent 的专属活动沙箱
# 强制设定在当前运行目录下的 "agent_workspace" 文件夹内
SANDBOX_DIR = Path("./agent_workspace").resolve()
# 启动时自动创建这个安全屋
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# 设定固定的知识库目录
KB_DIR = Path("./knowledge_base").resolve()
KB_DIR.mkdir(parents=True, exist_ok=True) # 如果没有会自动创建

# 定义允许读取的文件后缀白名单
ALLOWED_EXTENSIONS = {'.pdf', '.md', '.txt', '.csv'}

# 🌟 FAISS 向量硬盘持久化目录
FAISS_DB_DIR = Path("./embeddings").resolve()
FAISS_DB_DIR.mkdir(parents=True, exist_ok=True)

# 🌟 LRU 缓存配置：最多缓存 10 个文件的向量索引，防止内存泄漏
# 每个向量库约 10-50MB，10 个文件约占用 500MB 内存
MAX_FAISS_CACHE_SIZE = 10

# 定义一个专门存放记忆碎片的目录
MEMORY_DIR = Path("./memory").resolve()
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# 🌟 新增：长期记忆提取工具 (LTM)
USER_PROFILE_PATH = Path("./memory/user_profile.json").resolve()

# 创建一个锁文件
LOCK_PATH = Path("./memory/user_profile.json.lock").resolve()

# ==========================================
# 极客视觉核心：自定义回调拦截器
# ==========================================
class HackerMatrixCallback(BaseCallbackHandler):
    """拦截 Agent 的内部思考流，并用极其赛博朋克的方式打印到终端"""
    
    def on_agent_action(self, action, **kwargs):
        # 当 Agent 决定调用工具时触发
        # action.log 包含它的思考过程 (Thought)
        thought_text = action.log.split("Action:")[0].strip()
        
        # 打印绿色加粗的思考过程
        console.print(f"\n[bold green]▶ 核心思考协议接入...[/bold green]")
        console.print(f"[green dim]{thought_text}[/green dim]")
        
        # 打印亮蓝色的工具调用指令
        console.print(f"[bold cyan]⚡ 触发本地系统指令:[/bold cyan] [bold yellow]{action.tool}[/bold yellow]")
        console.print(f"[cyan dim]   载入参数: {action.tool_input}[/cyan dim]")

    def on_tool_end(self, output, **kwargs):
        # 当工具执行完毕，返回数据时触发
        # 截取前 150 个字符，营造一种数据流快速闪过的感觉
        snippet = str(output)[:150].replace('\n', ' ') + "..."
        console.print(f"[bold magenta]✔️ 数据流捕获成功:[/bold magenta] [magenta dim]{snippet}[/magenta dim]")

    def on_agent_finish(self, finish, **kwargs):
        # 最终任务完成时触发
        console.print("\n[bold green]▓▓▓▓▓▓▓▓ 任务执行完毕 ▓▓▓▓▓▓▓▓[/bold green]")

# ==========================================
# 插件 1：通过 yahoo 的标准接口查询美股、港股、A 股股价 (支持指定日期)
# ==========================================
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_universal_stock_price(ticker: str, date: str = None) -> str:
    """
    🌐 全球股票查价引擎（支持美股、A 股、港股）。
    只需传入用户提到的代码即可（例如：AAPL, 600519, 0700），底层会自动判断市场。
    - 参数 date (可选): 'YYYY-MM-DD'。未提供则默认返回最近交易日。
    """
    price_data = fetch_stock_price_raw(ticker, date)
    return (
        f"✅ {price_data['ticker']} ({price_data['date']}) - "
        f"开盘价：{price_data['open']}, 收盘价：{price_data['close']}"
    )
    

# ==========================================
# 插件 1.2：A 股 ETF 基金专用查询工具
# ==========================================
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_etf_price(etf_code: str, date: str = None) -> str:
    """
    🇨🇳 A 股 ETF 基金专用查价引擎（支持 akshare 和 yfinance 双数据源）。
    当用户查询 ETF 基金（如 513050、159915、510300 等）时优先使用此工具。
    - 参数 etf_code: 6 位 ETF 代码（如 '513050'）
    - 参数 date (可选): 'YYYY-MM-DD'。未提供则返回最近交易日数据。
    """
    price_data = fetch_etf_price_raw(etf_code, date)
    
    if price_data.get("source") == "akshare_spot":
        return (
            f"✅ ETF {etf_code} 实时行情 - 最新价：{price_data['current_price']} ({price_data['change_percent']}%)\n"
            f"开盘：{price_data['open']}, 最高：{price_data['high']}, 最低：{price_data['low']}, 昨收：{price_data['prev_close']}\n"
            f"成交量：{price_data['volume']}手，成交额：{price_data['amount']}万元"
        )
    else:
        return (
            f"✅ ETF {etf_code} ({price_data['date']}) - "
            f"开盘：{price_data['open']}, 收盘：{price_data['close']}, "
            f"最高：{price_data['high']}, 最低：{price_data['low']}, "
            f"成交量：{price_data['volume']}"
        )


# ==========================================
# 插件 1：绘图引擎
# ==========================================
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def draw_universal_stock_chart(ticker: str, days: int = 30) -> str:
    """
    🌐 全球股票走势绘图引擎（支持美股、A 股、港股，支持自定义时间跨度）。
    
    Args:
        ticker: 股票代码（如 AAPL, 600519, 0700）
        days: K 线图的时间跨度（天数）。默认 30 天。如果用户要求看'半年'请传 180，'一年'请传 365，'最近'或未指定则默认传 30。
    
    Returns:
        str: 生成结果与文件路径
    """
    chart_data = generate_kline_chart(ticker, SANDBOX_DIR, days)
    return (
        f"✅ {chart_data['ticker']} {days}天走势图生成完毕！文件名为：{chart_data['file_name']}。\n"
        f"【摘要】最高：{chart_data['max_price']}, 最低：{chart_data['min_price']}, 最新：{chart_data['latest_close']}。\n"
        f"🚨【强制语法】：必须严格使用 `![走势图](./{chart_data['file_name']})` 嵌入 Markdown 中！"
    )

# ==========================================
# 插件 2：代码搜索工具
# ==========================================
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_company_ticker(company_name: str) -> str:
    """
    当你不知道某家公司、产品或品牌的具体美股股票代码时，必须先使用此工具。
    输入公司或产品名称（如 'aws', '淘宝', '马斯克的公司'），它会联网搜索并返回相关信息以供你提取股票代码。
    """
    import requests
    
    try:
        query = f"{company_name} 股票代码 ticker symbol"
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        
        if not results:
            return f"未搜索到 {company_name} 的相关股票代码。"
        
        return str(results)
    except requests.exceptions.Timeout:
        return f"联网搜索超时：搜索 '{company_name}' 时超过 10 秒无响应，请稍后重试。"
    except requests.exceptions.ConnectionError:
        return f"网络连接失败：无法连接到搜索服务，请检查网络状态。"
    except requests.exceptions.HTTPError as e:
        return f"搜索服务返回错误：HTTP {e.response.status_code if hasattr(e, 'response') else 'ERROR'}。"
    except Exception as e:
        return f"联网搜索出错：{type(e).__name__} - {str(e)}"

# ==========================================
# 插件 3：读取本地文件
# ==========================================
@tool
def read_local_file(file_path: str) -> str:
    """
    当需要读取本地沙箱中的文件内容（如之前生成的报告）时调用此工具。
    输入参数为沙箱内的文件名或相对路径（例如：'report.md' 或 'data/info.txt'）。
    注意：出于安全限制，你只能读取沙箱(agent_workspace)内的文件。
    """
    try:
        # 1. 路径拼接与绝对路径解析
        target_path = (SANDBOX_DIR / file_path).resolve()
        
        # 2. 🌟 核心防御：使用 is_relative_to 替代 startswith
        # 这是 Python 3.9+ 提供的原生方法，它按层级严格判断，彻底杜绝平级恶意目录的绕过
        if not target_path.is_relative_to(SANDBOX_DIR):
            return "❌ 安全拦截：探测到越权操作！你试图读取沙箱外部的文件，已被系统拒绝。"

        # 3. 检查文件是否存在
        if not target_path.exists():
            return f"❌ 找不到文件: {target_path.name}"
            
        # 4. 安全执行读取
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return f"文件 {target_path.name} 的内容是:\n{content}"
        
    except FileNotFoundError:
        return f"❌ 文件不存在：{target_path.name}"
    except PermissionError:
        return f"❌ 权限不足：无法读取 {target_path.name}"
    except UnicodeDecodeError:
        return f"❌ 编码错误：{target_path.name} 不是有效的 UTF-8 文件"
    except Exception as e:
        return f"读取文件出错：{type(e).__name__} - {str(e)}"

# ==========================================
# 插件 4：写入本地文件
# ==========================================
@tool
def write_local_file(file_path: str, content: str) -> str:
    """
    🚨【强制交付通道】：
    当你被要求“写报告”、“生成分析”、“保存到本地”时，**绝对禁止**在聊天窗口直接输出 Markdown 文本！
    你必须且只能调用此工具，将完整排版好的 Markdown 内容作为 `content` 参数传入。
    输入参数 file_path 为目标文件名（例如：'report.md'）。
    """
    try:
        # 1. 路径拼接与绝对路径解析 (核心防御步 1)
        # 即使大模型传入类似 '../../隐藏目录/危险文件.txt' 的恶意路径，
        # .resolve() 也会在底层将其拉直，计算出真实的绝对路径。
        target_path = (SANDBOX_DIR / file_path).resolve()
        
        # 2. 越权判定 (核心防御步 2)
        # 检查解析后的最终真实路径，是不是以我们的沙箱目录为开头的
        # 如果不是，说明它用 ../ 成功逃逸到了上层目录，直接拦截！
        # 将 startswith 替换为底层的层级判定
        if not target_path.is_relative_to(SANDBOX_DIR):
            return "❌ 安全拦截：探测到越权操作！你试图将文件写入沙箱外部，已被系统拒绝。"

        # 3. 确保沙箱内的合法子目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 4. 安全执行写入
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"✅ 成功！报告已安全写入沙箱: {target_path}"
        
    except FileNotFoundError:
        return f"❌ 路径不存在：无法创建目录"
    except PermissionError:
        return f"❌ 权限不足：无法写入 {target_path}"
    except OSError as e:
        return f"❌ 系统错误：{type(e).__name__} - {str(e)}"
    except Exception as e:
        return f"写入文件出错：{type(e).__name__} - {str(e)}"

# ==========================================
# 🧠 底层向量加载引擎（L1/L2/L3 三级穿透架构）
# ==========================================
@lru_cache(maxsize=MAX_FAISS_CACHE_SIZE)
def _get_or_build_vectorstore(file_name: str, target_path_str: str, current_mtime: float):
    """
    带 LRU 缓存的向量库加载引擎，实现 L1 内存→L2 硬盘→L3 重建三级穿透。
    
    Args:
        file_name: 文件名（如 'report.pdf'）
        target_path_str: 文件绝对路径字符串
        current_mtime: 文件修改时间戳
    
    Returns:
        FAISS 向量库对象
    
    Note:
        - 利用 lru_cache 的 maxsize 限制内存占用（默认最多 10 个文件）
        - 利用 current_mtime 作为缓存 key 的一部分，文件修改后自动穿透失效
    """
    doc_cache_dir = FAISS_DB_DIR / f"{file_name}_vstore"
    meta_file = doc_cache_dir / "meta.json"
    
    try:
        embeddings = DashScopeEmbeddings(
            dashscope_api_key=embedding_key,
            model="text-embedding-v3",
        )
    except Exception as e:
        raise RuntimeError(f"向量模型初始化失败：{type(e).__name__} - {str(e)}")
    
    # ==========================================
    # 💾 尝试 L2 硬盘缓存
    # ==========================================
    if doc_cache_dir.exists() and meta_file.exists():
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            if not isinstance(meta, dict):
                raise TypeError("缓存元数据格式错误")
            
            if meta.get("mtime") == current_mtime:
                console.print(f"[bold cyan]💾 L2 命中 (硬盘):[/bold cyan] [cyan dim]加载 {file_name} 的持久化索引[/cyan dim]")
                vectorstore = FAISS.load_local(
                    str(doc_cache_dir),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                return vectorstore
        except (json.JSONDecodeError, TypeError) as e:
            console.print(f"[bold red]缓存元数据损坏 ({type(e).__name__})，准备降级重建[/bold red]")
        except Exception as e:
            console.print(f"[bold red]读取硬盘缓存失败 ({type(e).__name__})，准备降级重建：{str(e)}[/bold red]")
    
    # ==========================================
    # 🔄 L2 未命中：触发 L3 重建并持久化
    # ==========================================
    console.print(f"[bold blue]🔄 构建索引:[/bold blue] [blue dim]正在对 {file_name} 进行解析、向量化与持久化...[/blue dim]")
    
    target_path = Path(target_path_str)
    ext = target_path.suffix.lower()
    if ext == '.pdf':
        loader = PyPDFLoader(target_path_str)
    elif ext in ['.md', '.txt', '.csv']:
        loader = TextLoader(target_path_str, encoding='utf-8')
    else:
        raise ValueError(f"不支持的文件格式：{ext}")
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    raw_splits = text_splitter.split_documents(docs)
    splits = [s for s in raw_splits if s.page_content.strip()]
    
    if not splits:
        raise ValueError(f"文件 {file_name} 内容为空或无法提取有效文本")
    
    # 构建新的向量库
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 写入 L2 硬盘
    doc_cache_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(doc_cache_dir))
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({"mtime": current_mtime, "file_name": file_name}, f)
    
    console.print(f"[bold green]✅ 索引构建完成并已持久化到硬盘[/bold green]")
    return vectorstore


# ==========================================
# 插件 5：RAG 本地文档检索器 (L1 内存 + L2 硬盘 混合持久化架构)
# ==========================================
@tool
def analyze_local_document(file_name: str, query: str) -> str:
    """
    分析知识库中的文档（支持 PDF、Markdown、TXT 等）并回答问题。
    输入参数 file_name 只需要提供文件名（例如 'report.pdf' 或 'readme.md'），不要提供完整路径！
    """
    try:
        target_path = (KB_DIR / file_name).resolve()
        
        # 安全拦截
        if not target_path.is_relative_to(KB_DIR):
            return "❌ 安全拦截：你试图读取知识库以外的文件！"

        if not target_path.exists():
            return f"❌ 找不到文件：{file_name}。请先使用 list_kb_files 工具查看当前有哪些文件。"
        
        # 获取文件修改时间戳（用于 lru_cache 自动失效）
        current_mtime = os.path.getmtime(target_path)
        target_path_str = str(target_path)
        
        # 🚀 调用底层向量加载引擎（自动 L1/L2/L3 穿透）
        vectorstore = _get_or_build_vectorstore(file_name, target_path_str, current_mtime)
        
        # 执行检索操作
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        
        context = "\n---\n".join([doc.page_content for doc in relevant_docs])
        return f"✅ 从文档 {file_name} 中检索到以下核心信息：\n{context}\n\n请根据以上数据回答。"
        
    except json.JSONDecodeError:
        return f"❌ 文档元数据损坏：JSONDecodeError"
    except FileNotFoundError:
        return f"❌ 文档不存在：{file_name}"
    except ValueError as e:
        return f"❌ 数据错误：{e}"
    except RuntimeError as e:
        return f"❌ 系统错误：{e}"
    except Exception as e:
        return f"解析或检索文档出错：{type(e).__name__} - {str(e)}"

# ==========================================
# 插件 6：给 Agent 一双“眼睛”去查看知识库
# ==========================================
@tool
def list_kb_files() -> str:
    """
    当用户让你从知识库搜索，或者你不知道具体文件名时，必须先调用此工具！
    它会返回知识库文件夹下所有可用的文件列表。
    """
    try:
        # 扫描白名单内的所有文件
        files = [f.name for f in KB_DIR.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
        if not files:
            return "当前知识库文件夹为空，没有找到任何支持的文件。"
        return f"知识库中当前有以下文件可以读取:\n" + "\n".join(files)
    except FileNotFoundError:
        return "❌ 知识库目录不存在"
    except PermissionError:
        return "❌ 权限不足：无法访问知识库目录"
    except Exception as e:
        return f"读取目录出错：{type(e).__name__} - {str(e)}"

# ==========================================
# 插件 7：长期记忆提取
# ==========================================
# 工具 1：KV 状态机 (覆盖型)
@tool
def update_user_memory(key: str, value: str) -> str:
    """
    🚨【记忆更新指令】：
    用于记录或更新用户的状态、偏好、持仓快照。相同 key 会直接覆盖。
    - 参数 key: 记忆的分类标签，必须是简短明确的名词（例如："风险偏好"、"报告格式要求"）。
    - 参数 value: 具体的客观事实数据（例如："150 股，成本 150 美元"、"激进型"、"只看 Markdown 结论"）。
    注意：如果同一个 key 已经存在，新的 value 将直接【覆盖】旧数据！如果用户清仓了，你可以把 value 设置为 "已清仓" 或 "无"。
    
    🚨 【强制代码转换】（股票持仓专属红线）：
    当你要记录用户的具体股票持仓时，严禁使用自然语言公司名（如"苹果"、"腾讯"）作为 key！
    你必须先自行确认，或调用 `search_company_ticker` 查明其标准的股票代码（如 "AAPL", "0700.HK", "600519.SS"），然后**严格使用该标准代码作为 key** 写入记忆。
    例如：正确的调用是 key="AAPL", value="100 股，成本 150"，绝对不能是 key="苹果公司持仓"。
    
    🚨【持仓格式红线】：
    记录持仓时，`value` 中必须且只能按照『[中文公司名称/基金名称]，X 股，成本 Y』的格式记录！
    例如：`苹果公司，100 股，成本 150` 或 `纳斯达克 ETF，1000 股，成本 1.2`。
    绝对禁止省略中文名称！绝对禁止使用『买入价』、『单价』等同义词替换『成本』二字，否则底层计算引擎将无法识别！
    """
    try:
        # 初始化 JSON
        if not USER_PROFILE_PATH.exists():
            USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        
        # 🌟 核心：加锁！在 with 语句块内，其他任何企图读写这个文件的线程都会被阻塞等待
        with FileLock(LOCK_PATH, timeout=5): 
            # 1. 读入
            with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 2. 修改 (内存中)
            data[key] = value
            
            # 3. 写回
            with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return f"✅ 记忆已安全写入（加锁保护）：[{key}] -> '{value}'"
    except json.JSONDecodeError:
        return "❌ 记忆文件损坏：JSONDecodeError"
    except TimeoutError:
        return "❌ 文件锁超时：其他进程正在写入记忆"
    except Exception as e:
        return f"记忆写入失败：{type(e).__name__} - {str(e)}"

# 工具 2：事件流水账 (追加型) - 可选新增
@tool
def append_transaction_log(action: str, target: str, details: str) -> str:
    """
    🚨【交易日志指令】：
    仅当用户明确发生了一笔【交易动作】（如：买入、卖出、转账）时调用。
    它会像流水账一样把这笔操作追加到数据库中，绝对不会覆盖过去的历史。
    """
    try:
        log_path = Path("./memory/transaction_logs.jsonl")
        import time
        log_entry = json.dumps({
            "timestamp": time.time(),
            "action": action,     # 例如："买入"
            "target": target,     # 例如："苹果股票"
            "details": details    # 例如："100 股，成本 150"
        }, ensure_ascii=False)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        return "✅ 交易流水已追加记录。"
    except json.JSONDecodeError:
        return "❌ 日志序列化失败：JSONDecodeError"
    except FileNotFoundError:
        return "❌ 日志目录不存在"
    except Exception as e:
        return f"记录流水失败：{type(e).__name__} - {str(e)}"


# ==========================================
# 插件 8：个人持仓市值精确计算器
# ==========================================
@tool
def calculate_exact_portfolio_value() -> str:
    """
    💰【个人持仓市值精确计算器】（财务计算专属红线）：
    当用户询问自己的总资产、总市值、具体盈亏金额，或者要求盘点当前账户资金情况时，**必须且只能**调用此工具！
    **严禁自行心算或数学推演！**
    
    此工具会：
    1. 读取 ./memory/user_profile.json 中的持仓数据
    2. 调用底层估值引擎获取实时股价
    3. 精确计算总市值、总成本、今日盈亏
    
    Returns:
        str: 格式化的 Markdown 报告，包含总资产概览和各持仓明细表格
    """
    try:
        if not USER_PROFILE_PATH.exists():
            return "❌ 未找到持仓记忆文件，请先告知我您的持仓情况。"
        
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        if not user_data:
            return "❌ 持仓记忆为空，请先告知我您的持仓情况。"
        
        positions = parse_user_profile_to_positions(user_data)
        
        if not positions:
            return "❌ 未解析到有效持仓数据，请检查持仓记忆格式。"
        
        valuation = calculate_portfolio_valuation(positions)
        markdown_report = format_portfolio_report(valuation)
        
        return markdown_report
        
    except json.JSONDecodeError:
        return "❌ 持仓记忆文件损坏：JSONDecodeError"
    except Exception as e:
        return f"❌ 计算失败：{type(e).__name__} - {str(e)}"

# ==========================================
# 插件 9：主动触发盘后研报推送 (独立进程版)
# ==========================================
@tool
def trigger_daily_report() -> str:
    """
    🚨【研报推送专用指令】：
    当用户明确要求"现在给我发盘后日报"、"立刻推送研报"时，必须调用此工具。
    此工具已接入独立进程异步脱机引擎，你调用后只需告诉用户"任务已后台执行，请等待推送"即可。
    
    Returns:
        str: 任务提交结果，包含任务 ID 用于追踪
    """
    try:
        import subprocess
        import sys
        import uuid
        from pathlib import Path
        import json
        
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        status_dir = Path("./jobs/status").resolve()
        status_dir.mkdir(parents=True, exist_ok=True)
        
        initial_status = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "log_path": None
        }
        
        with open(status_dir / f"{job_id}.json", 'w', encoding='utf-8') as f:
            json.dump(initial_status, f, ensure_ascii=False, indent=2)
        
        console.print(f"[bold yellow]⏳ 正在将研报任务投递至独立进程 (Job ID: {job_id})...[/bold yellow]")
        
        process = subprocess.Popen(
            [sys.executable, "spawn_job.py", "--job-id", job_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        console.print(f"[bold green]✅ 进程已启动 | PID: {process.pid}[/bold green]")
        
        return (
            f"✅ 研报任务已成功挂载至后台独立进程！\n\n"
            f"**任务 ID**: <code>{job_id}</code>\n\n"
            f"您可以继续聊天或锁屏手机。大约 3~4 分钟后，研报将自动推送到您的屏幕。"
        )
        
    except Exception as e:
        return f"❌ 触发日报生成失败：{type(e).__name__} - {str(e)}"


# ==========================================
# 插件 10：查询研报任务状态
# ==========================================
@tool
def query_job_status(job_id: str) -> str:
    """
    查询研报任务的执行状态。
    
    Args:
        job_id: 任务唯一 ID（格式：job_YYYYMMDD_HHMMSS_abc123）
    
    Returns:
        str: 任务状态报告
    """
    try:
        from pathlib import Path
        import json
        
        status_file = Path(f"./jobs/status/{job_id}.json").resolve()
        
        if not status_file.exists():
            return f"❌ 未找到任务 {job_id}，请检查任务 ID 是否正确。"
        
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        
        status_map = {
            "pending": "⏳ 等待执行",
            "running": "🔄 正在执行中",
            "completed": "✅ 已完成",
            "failed": "❌ 执行失败"
        }
        
        result = [
            f"📊 **任务状态报告**",
            f"- **任务 ID**: `{job_id}`",
            f"- **当前状态**: {status_map.get(status.get('status'), '未知')}",
            f"- **创建时间**: {status.get('created_at', 'N/A')}",
        ]
        
        if status.get('started_at'):
            result.append(f"- **开始时间**: {status.get('started_at')}")
        
        if status.get('completed_at'):
            result.append(f"- **完成时间**: {status.get('completed_at')}")
        
        if status.get('error'):
            result.append(f"- **错误信息**: {status.get('error')}")
        
        if status.get('log_path'):
            result.append(f"- **日志路径**: {status.get('log_path')}")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"❌ 查询失败：{type(e).__name__} - {str(e)}"


# ==========================================
# 插件 11：自然语言智能盯盘预警创建器
# ==========================================
@tool
def create_price_alert(ticker: str, operator: str, target_price: float) -> str:
    """
    🚨【自然语言盯盘预警创建器】：
    当用户用自然语言要求"盯着"、"跌破"、"突破"、"涨过"某个具体价格时提醒他，必须调用此工具！
    你负责将用户的自然语言意图，转化为严谨的系统参数。
    
    Args:
        ticker: 股票代码（如 AAPL, 0700.HK, 513050.SS）。如果不确定，请先用 search_company_ticker 查询。
        operator: 必须严格输出 '<'（跌破/低于） 或 '>'（突破/高于/涨过）。
        target_price: 具体的触发价格（必须是纯数字）。
    """
    import json
    import os
    from pathlib import Path
    from datetime import datetime
    
    alerts_file = Path("./memory/alerts.json").resolve()
    alerts_file.parent.mkdir(parents=True, exist_ok=True)
    
    alerts = {}
    if alerts_file.exists():
        try:
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        except Exception:
            pass
            
    allowed_users = os.getenv("ALLOWED_TG_USERS", "").split(",")
    admin_id = allowed_users[0].strip() if allowed_users and allowed_users[0] else "default"
    
    if admin_id not in alerts:
        alerts[admin_id] = {}
        
    task_key = f"{ticker}_{operator}_{target_price}"
    alerts[admin_id][task_key] = {
        "ticker": ticker,
        "operator": operator,
        "target_price": target_price,
        "created_at": datetime.now().isoformat()
    }
    
    with open(alerts_file, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)
        
    return f"✅ 预警任务已安全挂载至底层引擎：当 {ticker} {operator} {target_price} 时将自动拦截并通知用户。"


tools = [get_universal_stock_price,
         get_etf_price,
         draw_universal_stock_chart,
         search_company_ticker,
         read_local_file, write_local_file,
         list_kb_files,
         analyze_local_document,
         update_user_memory,
         append_transaction_log,
         calculate_exact_portfolio_value,
         trigger_daily_report,
         query_job_status,
         create_price_alert]

# ==========================================
# 🧠 配置长效记忆引擎 (Long-Term Memory)
# ==========================================

def get_session_history(session_id: str):
    """
    带有滑动窗口截断机制的短期记忆引擎。
    """
    memory_file = str(MEMORY_DIR / f"{session_id}.json")
    history = FileChatMessageHistory(memory_file)
    
    # 🌟 核心省钱逻辑：滑动窗口截断
    # 如果对话超过 10 条（5次问答），我们就把更早的逐字稿清理掉，只保留最新的 10 条。
    # 那些重要的历史信息，已经被大模型用 remember_user_fact 存进 user_profile 里面了！
    if len(history.messages) > 10:
        kept_messages = history.messages[-10:]
        history.clear() # 清空臃肿的文件
        for msg in kept_messages:
            history.add_message(msg) # 把最新的 10 条写回去
            
    return history

def get_user_profile():
    """读取 KV 结构的长期记忆"""
    if not USER_PROFILE_PATH.exists():
        return "暂无长期记忆"
    try:
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            return "暂无长期记忆"
        # 转化为大模型易读的格式
        return "\n".join([f"- 【{k}】: {v}" for k, v in data.items()])
    except Exception as e:
        console.print(f"[dim]⚠️  读取长期记忆失败：{type(e).__name__}[/dim]")
        return "暂无长期记忆"

# 使用 ChatOpenAI 包装器，但把底层请求地址指向阿里云
llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=SecretStr(dashscope_key),
    base_url="https://coding.dashscope.aliyuncs.com/v1",
    temperature=0,
    timeout=90,
    max_retries=3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个极客风格的全栈量化分析师与系统助手。
     🕒 【系统物理时钟】：当前的真实现实时间是 {current_time}。你需要以此为绝对基准来理解用户的相对时间描述（如"今天"、"上周"、"昨天"），并判断当前所处的交易周期。
     🧠 【用户的长期记忆库】(以下是关于用户的客观事实，请在分析时主动结合使用){user_profile}。
     ==============================
    🔍 核心能力：
    - 遇到 ETF 基金查价（如 513050、159915 等 6 位数字代码），优先调用 `get_etf_price`；
    - 遇到股票查价，调用 `get_universal_stock_price`；
    - 遇到画图需求，调用 `draw_universal_stock_chart`。
    工具会在底层自动识别美股/A 股/港股，你无需操心市场后缀，直接传入用户给的代码即可。
    ==============================
    🚨【财务计算红线】（最高优先级）：
    当用户询问自己的总资产、总市值、具体盈亏金额，或者要求盘点当前账户资金情况时，
    **绝对禁止自行数学推演或心算！**
    **必须且只能调用 `calculate_exact_portfolio_value` 工具获取精确数据！**
    ==============================
    🚨 【记忆存储路由法则】（最高优先级判断逻辑）
    当你接收到用户的新信息时，你必须在脑海中进行分类，并严格调用对应的工具：

    1. 🎯 【状态与偏好】 -> 调用 `update_user_memory`
    - 触发条件：用户告知了当前持仓的总快照、个人投资偏好、习惯要求、人设设定。
    - 判断标准：这个信息是"排他"的，新的状态会使旧的状态失效。
    - 例子："我现在手里有 200 股特斯拉"、"以后别给我生成图表了"。

    2. 📜 【交易与事件】 -> 调用 `append_transaction_log`
    - 触发条件：用户告知了一笔具体的动作或历史发生过的事件。
    - 判断标准：它是流水账，不能覆盖。
    - 例子："我今天早上卖了 50 股苹果"、"我昨天把特斯拉清仓了"。

    3. 📚 【深度知识】 -> 调用 `write_local_file`
    - 触发条件：你为用户生成了深度的长篇分析、总结了某个行业的长文。
    - 判断标准：文字量极大，需要持久化保存为 Markdown 供日后 RAG 检索。

    4. 💬 【短期闲聊】 -> 不调用任何记忆工具！
    - 触发条件：随口的提问、查当前价格、简单的问答。
    - 判断标准：信息时效性极短，交给底层默认的短期滑动窗口记忆处理即可。
    ==============================
     工作流如下：
    1. 🔍 核心能力：遇到不知道的公司用 search_company_ticker，查本地资料用 analyze_local_document。
    2. ✍️ 智能输出调度（最高法则）：
       - ⚡ 轻量级问答：如果用户只是单纯询问价格或简单问题，请直接在终端简明扼要地回答。
       - 📝 盘后研报生成：当用户明确要求"生成报告"、"盘后研报"、"推送研报"时，**绝对禁止你自行搜集数据或进行财务核算！** 你必须且只能**立刻唯一**地调用 `trigger_daily_report` 工具，将任务移交给后台引擎。
       - 📚 自定义长文保存：只有当用户要求你写一篇*非盘后研报*的特定深度分析并保存时，才调用 `write_local_file`。
    
    3. 🖼️ 图文并茂：生成报告时，请务必先调用 draw_universal_stock_chart 生成走势图，并在传给 write_local_file 的 Markdown 内容中，使用 `![图表](./xxx.png)` 将图片嵌入。Telegram Bot 已支持将 .md 文件自动转换为 PDF 发送给用户（图片完整内嵌），因此请放心在报告中嵌入图表。
    4. 🧠 记忆系统：结合用户历史告知你的持仓情况或偏好进行解读。"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

# 提示：这里我把 verbose 改成了 False，这样终端里就不会打印大段的思考过程，更像真人在聊天
# 如果你想看它调用工具的底层细节，可以改回 True
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) 

# 使用 RunnableWithMessageHistory 包装原有的执行器
# 它会在每次调用前自动把 memory 里的历史塞进 {chat_history}，并在调用后把新对话存起来
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ==========================================
# 终端交互主循环 (REPL)
# ==========================================
if __name__ == "__main__":
    print("\n🤖 股票分析 Agent 已启动！(输入 'quit', 'exit' 或 '退出' 结束对话)")
    print("-" * 60)
    
    # 初始化高级会话（带内存历史记录）
    # 这样你不仅能左右移动光标修改错误，还能按“上/下方向键”调出上一轮问过的问题！
    session = PromptSession(history=InMemoryHistory())
    
    # 自定义一个好看的提示符样式（可选，让界面更有极客感）
    style = Style.from_dict({
        'prompt': 'ansicyan bold', # 提示符用青色加粗
    })

    while True:
        try:
            # 1. 使用高级 prompt 替代原生的 input()
            # 这里的输入体验将极其丝滑，支持所有快捷键和光标移动
            user_input = session.prompt('\n你: ', style=style)
            
            # 2. 设置退出条件
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("Agent: 再见！祝你投资顺利。")
                break
                
            # 防止输入空字符报错
            if not user_input.strip():
                continue
                
            # 3. 将输入发给带有记忆的 Agent
            response = agent_with_chat_history.invoke(
                {
                    "input": user_input,
                    # 🌟 每次对话前，动态读取并注入长期记忆！
                    "user_profile": get_user_profile(),
                    # 🌟 核心：每次用户按下回车时，动态获取当前精确时间并注入！
                    "current_time": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                },
                config={
                    "configurable": {"session_id": "terminal_session_01"},
                    "callbacks": [HackerMatrixCallback()] # 🌟 在这里挂载黑客视觉滤镜！
                }
            )
            
            # 4. 🌟 终极视觉渲染：支持 Markdown 结构化排版
            print() # 输出前补充一个空行，保持顶部的呼吸感
            
            # 顶部自适应边界线
            console.print(Rule("[bold cyan]SYS.RESPONSE[/bold cyan]", style="cyan"))
            
            # 核心：使用 Rich 的 Markdown 引擎进行渲染
            console.print(Markdown(response['output']))
            
            # 底部收尾边界线
            console.print(Rule("[dim cyan]EOF[/dim cyan]", style="cyan"))
            print() # 输出后补充空行
            
        except KeyboardInterrupt:
            # 捕捉 Ctrl+C，防止程序直接崩溃报错退出，而是优雅地中止当前输入
            print("\n[操作取消，按退出指令结束程序]")
            continue
        except EOFError:
            # 捕捉 Ctrl+D 优雅退出
            print("\nAgent: 再见！")
            break
        except Exception as e:
            print(f"\n[系统报错]: {str(e)}")