import os
import json
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import yfinance as yf
from langchain_core.tools import tool
# 使用openai 兼容千问
from langchain_openai import ChatOpenAI
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
from rich.panel import Panel
from langchain.callbacks.base import BaseCallbackHandler

# 初始化富文本控制台
console = Console()

# 在程序启动的最开始，调用 load_dotenv()
# 它会自动在当前目录寻找 .env 文件，并把里面的值载入到系统环境变量中
load_dotenv()

# 安全地获取 Key
# 使用 os.getenv() 获取，如果 .env 里没配，它会返回 None 而不是直接崩溃
dashscope_key = os.getenv("DASHSCOPE_API_KEY")

# 在程序刚启动时就检查关键依赖，如果没配 Key，立刻阻断并报错，而不是等跑了一半才死掉
if not dashscope_key:
    raise ValueError("❌ 致命错误：未在 .env 文件或环境变量中找到 DASHSCOPE_API_KEY！请检查配置。")

# 强行清除当前脚本的代理环境变量，强制直连
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

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

# 🌟 新增：FAISS 向量硬盘持久化目录
FAISS_DB_DIR = Path("./embeddings").resolve()
FAISS_DB_DIR.mkdir(parents=True, exist_ok=True)

# 🌟 新增：FAISS 向量库全局内存缓存池
# 字典结构: { "文件绝对路径": {"mtime": 12345678.9, "vectorstore": <FAISS_Object>} }
FAISS_CACHE = {}

# 定义一个专门存放记忆碎片的目录
MEMORY_DIR = Path("./memory").resolve()
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# 🌟 新增：长期记忆提取工具 (LTM)
USER_PROFILE_PATH = Path("./memory/user_profile.json").resolve()

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
# 插件 1：通过yahoo的标准接口查询美股股价
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """输入美股代码（如 AAPL, MSFT），返回该股票最近一个交易日的开盘价和收盘价。"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return f"未找到 {ticker} 的数据"
        open_p = round(float(hist['Open'].iloc[0]), 2)
        close_p = round(float(hist['Close'].iloc[0]), 2)
        return f"{ticker} 最近交易日数据 - 开盘价: {open_p}, 收盘价: {close_p}"
    except Exception as e:
        return f"查询出错: {str(e)}"
    
# ==========================================
# 插件 1.5：K线图与 30 天走势可视化
# ==========================================
@tool
def draw_stock_chart(ticker: str) -> str:
    """
    获取指定美股代码（如 AAPL, MSFT）过去 1 个月的历史数据，绘制专业的 K线走势图与成交量，
    并将图片安全保存在本地沙箱中。
    当你需要为分析报告增加可视化图表，或者用户要求查看历史走势时，调用此工具。
    """
    try:
        stock = yf.Ticker(ticker)
        # 获取过去 1 个月的数据（包含 Open, High, Low, Close, Volume）
        hist = stock.history(period="1mo")
        if hist.empty:
            return f"❌ 未找到 {ticker} 的历史数据，无法绘图。"

        # 🌟 核心修复：移除 datetime 时间戳，使用确定性的纯粹命名！
        # 这样大模型绝对不会再拼错图片路径，且能自动覆盖旧图，保持沙箱整洁
        chart_filename = f"{ticker}_30d_chart.png"
        
        # 将图片路径强制锁定在沙箱目录内 (复用我们之前的防逃逸安全屋)
        chart_path = (SANDBOX_DIR / chart_filename).resolve()

        # 核心绘图逻辑：使用 mpf 画出带均线和成交量的雅虎风格 K线图
        mpf.plot(
            hist, 
            type='candle',       # K线图模式
            volume=True,         # 显示底部成交量
            style='yahoo',       # 雅虎财经配色风格 (红绿柱)
            title=f"{ticker} 30-Day Trend", 
            mav=(5, 10),         # 添加 5日和 10日移动均线
            savefig=str(chart_path) # 直接保存到沙箱，不弹窗
        )

        # 提取极值，作为 prompt 补充信息传给 LLM
        max_price = round(hist['High'].max(), 2)
        min_price = round(hist['Low'].min(), 2)
        latest_close = round(hist['Close'].iloc[-1], 2)
        
        return (
            f"✅ {ticker} 的30天K线图已成功生成！文件名为：{chart_filename}。\n"
            f"【统计摘要】最高价: {max_price}, 最低价: {min_price}, 最新价: {latest_close}。\n"
            f"🚨【强制语法】：在你马上要生成的 Markdown 报告中，必须严格使用 `![{ticker}走势图](./{chart_filename})` 插入此图片，一个字都不能改！"
        )
    except Exception as e:
        return f"绘制图表出错: {str(e)}"

# ==========================================
# 插件 2：代码搜索工具
# ==========================================
@tool
def search_company_ticker(company_name: str) -> str:
    """
    当你不知道某家公司、产品或品牌的具体美股股票代码时，必须先使用此工具。
    输入公司或产品名称（如 'aws', '淘宝', '马斯克的公司'），它会联网搜索并返回相关信息以供你提取股票代码。
    """
    try:
        # 自动构造搜索词，抓取前 3 条网页摘要
        query = f"{company_name} stock ticker symbol 美股代码"
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"未搜索到 {company_name} 的相关股票代码。"
        
        # 将搜索到的网页摘要直接扔给大模型，它的“大脑”会自动从里面提取出正确的字母代码
        return str(results)
    except Exception as e:
        return f"联网搜索出错: {str(e)}"

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
        
    except Exception as e:
        return f"读取文件出错: {str(e)}"

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
        
    except Exception as e:
        return f"写入文件出错: {str(e)}"

# ==========================================
# 插件 5：RAG 本地文档检索器 (L1内存 + L2硬盘 混合持久化架构)
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
            return f"❌ 找不到文件: {file_name}。请先使用 list_kb_files 工具查看当前有哪些文件。"
            
        current_mtime = os.path.getmtime(target_path)
        target_path_str = str(target_path)
        
        # 为该文件计算专属的硬盘缓存目录名
        doc_cache_dir = FAISS_DB_DIR / f"{file_name}_vstore"
        meta_file = doc_cache_dir / "meta.json"
        
        embeddings = DashScopeEmbeddings(
            dashscope_api_key=dashscope_key,
            model="text-embedding-v3", 
        )
        
        # ==========================================
        # ⚡ 检查 L1 缓存 (内存)
        # ==========================================
        if target_path_str in FAISS_CACHE and FAISS_CACHE[target_path_str]["mtime"] == current_mtime:
            console.print(f"[bold yellow]⚡ L1 命中 (内存):[/bold yellow] [yellow dim]极速复用 {file_name} 的向量索引[/yellow dim]")
            vectorstore = FAISS_CACHE[target_path_str]["vectorstore"]
            
        else:
            # ==========================================
            # 💾 检查 L2 缓存 (硬盘)
            # ==========================================
            loaded_from_disk = False
            if doc_cache_dir.exists() and meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    # 校验硬盘缓存的时间戳是否与文件当前时间一致
                    if meta.get("mtime") == current_mtime:
                        console.print(f"[bold cyan]💾 L2 命中 (硬盘):[/bold cyan] [cyan dim]加载 {file_name} 的持久化索引并写回内存[/cyan dim]")
                        # 注意：allow_dangerous_deserialization=True 是必须的，因为我们要信任自己本地生成的 pickle 文件
                        vectorstore = FAISS.load_local(
                            str(doc_cache_dir), 
                            embeddings, 
                            allow_dangerous_deserialization=True 
                        )
                        # 反向预热 L1 内存池
                        FAISS_CACHE[target_path_str] = {"mtime": current_mtime, "vectorstore": vectorstore}
                        loaded_from_disk = True
                except Exception as e:
                    console.print(f"[bold red]读取硬盘缓存失败，准备降级重建: {str(e)}[/bold red]")
            
            # ==========================================
            # 🔄 均未命中 (或文件被修改)：触发 L3 重建并穿透写入
            # ==========================================
            if not loaded_from_disk:
                console.print(f"[bold blue]🔄 构建索引:[/bold blue] [blue dim]正在对 {file_name} 进行解析、向量化与持久化...[/blue dim]")
                
                ext = target_path.suffix.lower()
                if ext == '.pdf':
                    loader = PyPDFLoader(target_path_str)
                elif ext in ['.md', '.txt', '.csv']:
                    loader = TextLoader(target_path_str, encoding='utf-8')
                else:
                    return f"❌ 不支持的文件格式: {ext}。目前支持 {ALLOWED_EXTENSIONS}"

                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                raw_splits = text_splitter.split_documents(docs)
                splits = [s for s in raw_splits if s.page_content.strip()]
                
                if not splits:
                    return f"❌ 文件 {file_name} 内容为空，或者无法提取有效文本。"
                
                # 构建新的向量库
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                # 写入 L1 内存
                FAISS_CACHE[target_path_str] = {"mtime": current_mtime, "vectorstore": vectorstore}
                
                # 写入 L2 硬盘
                doc_cache_dir.mkdir(parents=True, exist_ok=True)
                vectorstore.save_local(str(doc_cache_dir))
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump({"mtime": current_mtime, "file_name": file_name}, f)

        # 执行真正的检索操作
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        
        context = "\n---\n".join([doc.page_content for doc in relevant_docs])
        return f"✅ 从文档 {file_name} 中检索到以下核心信息：\n{context}\n\n请根据以上数据回答。"
        
    except Exception as e:
        return f"解析或检索文档出错: {str(e)}"

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
    except Exception as e:
        return f"读取目录出错: {str(e)}"

# ==========================================
# 插件 7：长期记忆提取
# ==========================================
@tool
def remember_user_fact(fact: str) -> str:
    """
    🚨【记忆写入指令】：
    当你得知关于用户的关键信息（如：持仓情况、成本价、投资偏好、个人习惯等）时，必须调用此工具。
    输入参数 fact 是一句简短的客观事实描述，例如："用户持有 100 股 TSLA" 或 "用户不喜欢看长篇大论"。
    """
    try:
        # 确保文件存在
        if not USER_PROFILE_PATH.exists():
            USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
                json.dump({"facts": []}, f)
                
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 防止重复写入
        if fact not in data["facts"]:
            data["facts"].append(fact)
            with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return f"✅ 长期记忆已更新：{fact}"
        return "该记忆已存在。"
    except Exception as e:
        return f"记忆写入失败: {str(e)}"

tools = [get_stock_price, draw_stock_chart, search_company_ticker, read_local_file, write_local_file, list_kb_files, analyze_local_document, remember_user_fact]

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
    """读取用户长期记忆核心，转化为字符串注入 Prompt"""
    if not USER_PROFILE_PATH.exists():
        return "暂无长期记忆"
    try:
        with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data.get("facts"):
            return "暂无长期记忆"
        return "\n".join([f"- {fact}" for fact in data["facts"]])
    except:
        return "暂无长期记忆"

# 使用 ChatOpenAI 包装器，但把底层请求地址指向阿里云
llm = ChatOpenAI(
    model="qwen3.5-plus", # 强烈推荐用 qwen-max，处理复杂逻辑和多工具路由最稳
    api_key=dashscope_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 核心：指向阿里兼容接口
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个极客风格的全栈量化分析师与系统助手。
     🧠 【用户的长期记忆库】(以下是关于用户的客观事实，请在分析时主动结合使用){user_profile}。
     工作流如下：
    1. 🔍 核心能力：遇到不知道的公司用 search_company_ticker，查最新价格用 get_stock_price，查30天走势并画图用 draw_stock_chart，查本地资料用 analyze_local_document。
    2. ✍️ 智能输出调度（最高法则）：
       - ⚡ 轻量级问答：如果用户只是单纯询问价格或简单问题，请直接在终端简明扼要地回答，绝对不要调用 write_local_file。
       - 📝 深度报告生成：当用户要求“生成报告”、“保存到本地”、“写研报”时，你必须整合分析。
       
    🚨【绝对红线指令 - 报告怎么写】：
    如果你判断当前任务需要生成报告，你**严禁**在最终的终端回复（Final Answer）中直接输出报告的 Markdown 文本！
    你**必须且只能**将写好的整篇 Markdown 内容作为 `content` 参数，调用 `write_local_file` 工具保存！
    终端最终只需冷酷地汇报一句：“✅ 任务执行完毕。深度分析报告已生成，本地路径为：xxx”。
    
    3. 🖼️ 图文并茂：生成报告时，请务必先调用 draw_stock_chart 生成走势图，并在传给 write_local_file 的 Markdown 内容中，使用 `![图表](./xxx.png)` 将图片嵌入。
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
                    "user_profile": get_user_profile() # 🌟 每次对话前，动态读取并注入长期记忆！
                },
                config={
                    "configurable": {"session_id": "terminal_session_01"},
                    "callbacks": [HackerMatrixCallback()] # 🌟 在这里挂载黑客视觉滤镜！
                }
            )
            
            # 4. 用 Rich Panel 打印 Agent 的最终简短回复
            console.print(Panel(
                response['output'], 
                title="[bold cyan]SYS.RESPONSE[/bold cyan]", 
                border_style="cyan"
            ))
            
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