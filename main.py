import os
import json
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import mplfinance as mpf
import akshare as ak
from datetime import datetime, timedelta
from filelock import FileLock
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
from rich.markdown import Markdown   # 🌟 新增：Markdown 渲染引擎
from rich.rule import Rule           # 🌟 新增：自适应分隔线组件
from langchain.callbacks.base import BaseCallbackHandler
#添加超时处理逻辑
from tenacity import retry, stop_after_attempt, wait_exponential

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
# 插件 1：通过yahoo的标准接口查询美股股价 (支持指定日期)
# ==========================================
@tool
def get_stock_price(ticker: str, date: str = None) -> str:
    """
    输入美股代码（如 AAPL, MSFT），返回该股票的开盘价和收盘价。
    - 参数 ticker: 美股代码。
    - 参数 date (可选): 指定查询的日期，格式必须为 'YYYY-MM-DD'（例如 '2023-10-25'）。如果未提供此参数，则默认返回最近一个交易日的数据。
    """
    try:
        import yfinance as yf # 确保在作用域内
        stock = yf.Ticker(ticker)
        
        if date:
            # 如果提供了具体日期，解析它并计算下一天，以满足 yfinance 的区间查询要求
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
                next_date = target_date + timedelta(days=1)
                
                start_str = target_date.strftime("%Y-%m-%d")
                end_str = next_date.strftime("%Y-%m-%d")
                
                # 查询特定区间
                hist = stock.history(start=start_str, end=end_str)
                date_label = date
            except ValueError:
                return "❌ 查询出错：日期格式不正确。请大模型务必使用 'YYYY-MM-DD' 格式重试。"
        else:
            # 未提供日期，默认查询最近 1 天
            hist = stock.history(period="1d")
            date_label = "最近交易日"
            
        if hist.empty:
            return f"❌ 未找到 {ticker} 在 {date_label} 的数据（可能该日期为周末/节假日非交易日，或者股票代码错误）。"
            
        # 提取数据
        open_p = round(float(hist['Open'].iloc[0]), 2)
        close_p = round(float(hist['Close'].iloc[0]), 2)
        
        # 获取实际返回数据的日期（防止时区或 API 截断问题导致日期对不上）
        actual_date = hist.index[0].strftime("%Y-%m-%d")
        
        return f"✅ {ticker} 在 {actual_date} 的数据 - 开盘价: {open_p}, 收盘价: {close_p}"
        
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
# 插件 1.6：港股市场查价引擎 (带有自动格式化装甲)
# ==========================================
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_hk_stock_price(ticker: str, date: str = None) -> str:
    """
    🚨 专用于查询港股（香港股市）的股价。
    输入参数 ticker 必须是港股的数字代码（如 700, 9988, 3690）。
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # 🌟 核心防御：大模型可能传 "700"、"0700" 或 "0700.HK"，统一清洗
        ticker_num = ''.join(filter(str.isdigit, ticker))
        if not ticker_num:
            return f"❌ 港股代码格式错误：{ticker}。"
            
        # 自动补齐 4 位并加上 yfinance 识别的 .HK 后缀
        formatted_ticker = f"{ticker_num.zfill(4)}.HK"
        stock = yf.Ticker(formatted_ticker)
        
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
                next_date = target_date + timedelta(days=1)
                hist = stock.history(start=target_date.strftime("%Y-%m-%d"), end=next_date.strftime("%Y-%m-%d"))
                date_label = date
            except ValueError:
                return "❌ 查询出错：日期格式不正确。"
        else:
            hist = stock.history(period="1d")
            date_label = "最近交易日"
            
        if hist.empty:
            return f"❌ 未找到港股 {formatted_ticker} 在 {date_label} 的数据（可能为非交易日）。"
            
        open_p = round(float(hist['Open'].iloc[0]), 2)
        close_p = round(float(hist['Close'].iloc[0]), 2)
        actual_date = hist.index[0].strftime("%Y-%m-%d")
        
        return f"✅ 港股 {formatted_ticker} 在 {actual_date} 的数据 - 开盘价: {open_p}, 收盘价: {close_p}"
    except Exception as e:
        return f"查询港股出错: {str(e)}"

# ==========================================
# 插件 1.7：港股专属 K 线视觉渲染器
# ==========================================
@tool
def draw_hk_stock_chart(ticker: str) -> str:
    """
    🚨 专用于绘制港股（香港股市）的 30 天走势图。
    输入参数 ticker 必须是港股的数字代码（如 700, 9988）。
    """
    try:
        import yfinance as yf
        import mplfinance as mpf
        from datetime import datetime
        
        # 清洗并格式化代码
        ticker_num = ''.join(filter(str.isdigit, ticker))
        if not ticker_num:
            return f"❌ 港股代码格式错误：{ticker}。"
            
        formatted_ticker = f"{ticker_num.zfill(4)}.HK"
        stock = yf.Ticker(formatted_ticker)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return f"❌ 未找到港股 {formatted_ticker} 的历史数据，无法绘图。"
            
        # 生成确定性的文件名
        chart_filename = f"HK_{ticker_num}_30d_chart.png"
        chart_path = (SANDBOX_DIR / chart_filename).resolve()
        
        # 绘图逻辑
        mpf.plot(
            hist, type='candle', volume=True, style='yahoo',
            title=f"HK-Share {formatted_ticker} 30-Day Trend", mav=(5, 10),
            savefig=str(chart_path)
        )
        
        max_price = round(hist['High'].max(), 2)
        min_price = round(hist['Low'].min(), 2)
        latest_close = round(hist['Close'].iloc[-1], 2)
        
        return (
            f"✅ 港股 {formatted_ticker} 30天K线图生成完毕！文件名为：{chart_filename}。\n"
            f"【摘要】最高: {max_price}, 最低: {min_price}, 最新: {latest_close}。\n"
            f"🚨【强制语法】：必须严格使用 `![{ticker_num}走势图](./{chart_filename})` 嵌入 Markdown 中！"
        )
    except Exception as e:
        return f"港股绘图出错: {str(e)}"

# ==========================================
# 插件 1.8：A 股市场查价引擎 (基于 AkShare)
# ==========================================
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_ashare_price(ticker: str, date: str = None) -> str:
    """
    🚨 专用于查询中国 A 股的股价。
    输入参数 ticker 必须是 A股的 6 位纯数字代码（如 600519, 000001）。
    - 参数 date (可选): 指定查询日期 'YYYY-MM-DD'。未提供则默认返回最近一个交易日的数据。
    """
    try:
        # 清洗 ticker，强行剥离出纯数字
        ticker_num = ''.join(filter(str.isdigit, ticker))
        if len(ticker_num) != 6:
            return f"❌ A股代码格式错误：{ticker}，必须是 6 位纯数字代码。"
        
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
                start_date = end_date = target_date
                date_label = date
            except ValueError:
                return "❌ 日期格式不正确。请使用 'YYYY-MM-DD' 格式。"
        else:
            # 默认获取过去 7 天的数据，取最后一条确保能抓到最新的交易日
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
            date_label = "最近交易日"
            
        # 调用 AkShare 获取前复权日K线数据
        df = ak.stock_zh_a_hist(symbol=ticker_num, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df.empty:
            return f"❌ 未找到 A股 {ticker_num} 在 {date_label} 的数据（可能为非交易日或代码错误）。"
            
        latest_data = df.iloc[-1]
        open_p = round(float(latest_data['开盘']), 2)
        close_p = round(float(latest_data['收盘']), 2)
        actual_date = str(latest_data['日期'])[:10]
        
        return f"✅ A股 {ticker_num} 在 {actual_date} 的数据 - 开盘价: {open_p}, 收盘价: {close_p}"
        
    except Exception as e:
        return f"查询A股出错: {str(e)}"

# ==========================================
# 插件 1.9：A 股专属 K 线视觉渲染器
# ==========================================
@tool
def draw_ashare_chart(ticker: str) -> str:
    """
    🚨 专用于绘制中国 A 股的 30 天走势图。
    输入参数 ticker 必须是 6 位纯数字代码（如 600519）。
    """
    try:
        ticker_num = ''.join(filter(str.isdigit, ticker))
        if len(ticker_num) != 6:
            return f"❌ A股代码格式错误：{ticker}，必须是 6 位纯数字。"
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=40) # 多取几天保证凑够一整个月的交易日
        
        df = ak.stock_zh_a_hist(symbol=ticker_num, period="daily", start_date=start_date.strftime("%Y%m%d"), end_date=end_date.strftime("%Y%m%d"), adjust="qfq")
        
        if df.empty:
            return f"❌ 未找到 A股 {ticker_num} 的历史数据，无法绘图。"
            
        # 🌟 核心适配器：将 AkShare 的中文列名翻译成 mplfinance 识别的标准英文列名
        df = df.rename(columns={
            '日期': 'Date', '开盘': 'Open', '最高': 'High', 
            '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        chart_filename = f"A_{ticker_num}_30d_chart.png"
        chart_path = (SANDBOX_DIR / chart_filename).resolve()
        
        # 绘图逻辑
        mpf.plot(
            df, type='candle', volume=True, style='yahoo',
            title=f"A-Share {ticker_num} 30-Day Trend", mav=(5, 10),
            savefig=str(chart_path)
        )
        
        max_price = round(df['High'].max(), 2)
        min_price = round(df['Low'].min(), 2)
        latest_close = round(df['Close'].iloc[-1], 2)
        
        return (
            f"✅ A股 {ticker_num} 30天K线图生成完毕！文件名为：{chart_filename}。\n"
            f"【摘要】最高: {max_price}, 最低: {min_price}, 最新: {latest_close}。\n"
            f"🚨【强制语法】：必须严格使用 `![{ticker_num}走势图](./{chart_filename})` 嵌入 Markdown 中！"
        )
    except Exception as e:
        return f"A股绘图出错: {str(e)}"

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
    try:
        # 自动构造搜索词，抓取前 3 条网页摘要
        query = f"{company_name} 股票代码 ticker symbol"
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
# 工具 1：KV 状态机 (覆盖型)
@tool
def update_user_memory(key: str, value: str) -> str:
    """
    🚨【记忆更新指令】：
    用于记录或更新用户的状态、偏好、持仓快照。相同 key 会直接覆盖。
    - 参数 key: 记忆的分类标签，必须是简短明确的名词（例如："苹果公司持仓"、"风险偏好"、"报告格式要求"）。
    - 参数 value: 具体的客观事实数据（例如："150股，成本150美元"、"激进型"、"只看Markdown结论"）。
    注意：如果同一个 key 已经存在，新的 value 将直接【覆盖】旧数据！如果用户清仓了，你可以把 value 设置为 "已清仓" 或 "无"。
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
    except Exception as e:
        return f"记忆写入失败: {str(e)}"

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
            "details": details    # 例如："100股，成本150"
        }, ensure_ascii=False)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        return "✅ 交易流水已追加记录。"
    except Exception as e:
        return f"记录流水失败: {str(e)}"

tools = [get_stock_price,
         draw_stock_chart,
         search_company_ticker,
         read_local_file, write_local_file,
         list_kb_files,
         analyze_local_document,
         update_user_memory,
         append_transaction_log,
         get_ashare_price,
         draw_ashare_chart,
         get_hk_stock_price,
         draw_hk_stock_chart]

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
    except:
        return "暂无长期记忆"

# 使用 ChatOpenAI 包装器，但把底层请求地址指向阿里云
llm = ChatOpenAI(
    model="qwen3.5-plus", # 强烈推荐用 qwen-max，处理复杂逻辑和多工具路由最稳
    api_key=dashscope_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 核心：指向阿里兼容接口
    temperature=0,
    # 🌟 新增：配置请求超时与底层自动重试机制
    request_timeout=45,  # 设定 45 秒硬性超时阈值
    max_retries=3        # 遇到 502/504 等网络波动自动重试 3 次
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个极客风格的全栈量化分析师与系统助手。
     🕒 【系统物理时钟】：当前的真实现实时间是 {current_time}。你需要以此为绝对基准来理解用户的相对时间描述（如“今天”、“上周”、“昨天”），并判断当前所处的交易周期。
     🧠 【用户的长期记忆库】(以下是关于用户的客观事实，请在分析时主动结合使用){user_profile}。
     ==============================
    🚨 【跨国股票市场路由法则】（极其重要！）
    当用户询问股票数据或图表时，你必须根据股票所属市场，精确路由给对应的工具链：
    - 🇺🇸 【美股市场】（如 苹果/AAPL, 微软/MSFT, 英伟达/NVDA）：必须调用 `get_stock_price` 和 `draw_stock_chart`。
    - 🇨🇳 【A股市场】（如 贵州茅台/600519, 宁德时代/300750, 比亚迪）：必须提取出 **6位纯数字代码**，并调用专用的 `get_ashare_price`（查价）和 `draw_ashare_chart`（画图）。
    - 🇭🇰 【港股市场】（如 腾讯/0700, 阿里/9988, 美团/3690）：必须提取出 **数字代码**，并调用专用的 `get_hk_stock_price`（查价）和 `draw_hk_stock_chart`（画图）。
    ==============================
    🚨 【记忆存储路由法则】（最高优先级判断逻辑）
    当你接收到用户的新信息时，你必须在脑海中进行分类，并严格调用对应的工具：

    1. 🎯 【状态与偏好】 -> 调用 `update_user_memory`
    - 触发条件：用户告知了当前持仓的总快照、个人投资偏好、习惯要求、人设设定。
    - 判断标准：这个信息是“排他”的，新的状态会使旧的状态失效。
    - 例子：“我现在手里有200股特斯拉”、“以后别给我生成图表了”。

    2. 📜 【交易与事件】 -> 调用 `append_transaction_log`
    - 触发条件：用户告知了一笔具体的动作或历史发生过的事件。
    - 判断标准：它是流水账，不能覆盖。
    - 例子：“我今天早上卖了50股苹果”、“我昨天把特斯拉清仓了”。

    3. 📚 【深度知识】 -> 调用 `write_local_file`
    - 触发条件：你为用户生成了深度的长篇分析、总结了某个行业的长文。
    - 判断标准：文字量极大，需要持久化保存为 Markdown 供日后 RAG 检索。

    4. 💬 【短期闲聊】 -> 不调用任何记忆工具！
    - 触发条件：随口的提问、查当前价格、简单的问答。
    - 判断标准：信息时效性极短，交给底层默认的短期滑动窗口记忆处理即可。
    ==============================
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