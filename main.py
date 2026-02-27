import os
from dotenv import load_dotenv
from pathlib import Path
import yfinance as yf
from langchain_core.tools import tool
# 使用openai 兼容千问
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS
# 新增：用于管理记忆的模块
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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
    当需要读取本地文件内容时调用此工具。
    输入参数为文件的绝对路径或相对路径（例如：'config.json' 或 '/Users/xxx/data.txt'）。
    """
    try:
        if not os.path.exists(file_path):
            return f"错误：找不到文件 {file_path}"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件 {file_path} 的内容是:\n{content}"
    except Exception as e:
        return f"读取文件出错: {str(e)}"

# ==========================================
# 插件 4：写入本地文件
# ==========================================
@tool
def write_local_file(file_path: str, content: str) -> str:
    """
    当需要把文本、报告或代码保存到本地计算机时调用此工具。
    输入参数为目标文件名或相对路径（例如：'report.md' 或 'data/info.txt'）。
    注意：出于安全限制，你只能将文件写入到分配给你的工作区内，请直接提供文件名即可。
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
# 插件 5：RAG 本地文档检索器
# ==========================================
@tool
def analyze_local_document(file_name: str, query: str) -> str:
    """
    分析知识库中的文档（支持 PDF、Markdown、TXT 等）并回答问题。
    输入参数 file_name 只需要提供文件名（例如 'report.pdf' 或 'readme.md'），不要提供完整路径！
    """
    try:
        target_path = (KB_DIR / file_name).resolve()
        
        if not str(target_path).startswith(str(KB_DIR)):
            return "❌ 安全拦截：你试图读取知识库以外的文件！"

        if not target_path.exists():
            return f"❌ 找不到文件: {file_name}。请先使用 list_kb_files 工具查看当前有哪些文件。"
            
        # 🌟 核心升级：根据后缀名动态分配加载器
        ext = target_path.suffix.lower()
        if ext == '.pdf':
            loader = PyPDFLoader(str(target_path))
        elif ext in ['.md', '.txt', '.csv']:
            # 对于纯文本，强制使用 utf-8 编码读取，防止中文乱码
            loader = TextLoader(str(target_path), encoding='utf-8')
        else:
            return f"❌ 不支持的文件格式: {ext}。目前支持 {ALLOWED_EXTENSIONS}"

        # 加载文档
        docs = loader.load()
        
        # 数据切块 (后续逻辑完全保持不变)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        raw_splits = text_splitter.split_documents(docs)
        splits = [s for s in raw_splits if s.page_content.strip()]
        
        if not splits:
            return f"❌ 文件 {file_name} 内容为空，或者无法提取有效文本。"
        
        # 使用你跑通的 DashScope 原生向量接口
        embeddings = DashScopeEmbeddings(
            dashscope_api_key=dashscope_key,
            model="text-embedding-v3", 
        )
        
        vectorstore = FAISS.from_documents(splits, embeddings)
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

tools = [get_stock_price, search_company_ticker, read_local_file, write_local_file, list_kb_files, analyze_local_document]

# 使用 ChatOpenAI 包装器，但把底层请求地址指向阿里云
llm = ChatOpenAI(
    model="qwen-max", # 强烈推荐用 qwen-max，处理复杂逻辑和多工具路由最稳
    api_key=dashscope_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 核心：指向阿里兼容接口
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个极客风格的全栈量化分析师与系统助手。工作流如下：
    1. 信息获取：遇到不知道的公司用 search_company_ticker，查价格用 get_stock_price，查本地资料用 analyze_local_document。
    2. 【最高优先级指令】：你的所有分析任务，最终都**必须**生成一份排版精美的 Markdown 报告，并调用 write_local_file 工具将其保存在本地沙箱中（文件名建议使用英文或拼音，如 report_xxx.md）。
    3. 终端回复：文件保存成功后，在终端中**只需要**用极客的口吻简短汇报一句：“分析报告已生成，路径为：xxx”，不要在终端里长篇大论。"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

# 提示：这里我把 verbose 改成了 False，这样终端里就不会打印大段的思考过程，更像真人在聊天
# 如果你想看它调用工具的底层细节，可以改回 True
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) 

# ==========================================
# 配置记忆引擎
# ==========================================
# 在内存中开辟一块空间存储对话历史
memory = InMemoryChatMessageHistory()

def get_session_history(session_id: str):
    return memory

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
                {"input": user_input},
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