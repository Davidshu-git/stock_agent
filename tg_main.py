import ast
import os
import re
import time
import logging
import asyncio
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message, BotCommand, Bot
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv

# 类型提示
from typing import List

from main import SANDBOX_DIR, KB_DIR

# 🚀 新视觉引擎依赖
import markdown
from playwright.async_api import async_playwright

# 🌟 LangChain 异步回调核心依赖
from langchain_core.callbacks import AsyncCallbackHandler

# 🌟 无缝引入咱们精心打磨的底层 Agent 引擎
from main import agent_with_chat_history, get_user_profile
from valuation_engine import fetch_stock_price_raw



# 🛡️ 安全兜底：加载 .env 文件
load_dotenv()

# 🔐 从环境变量获取 Token
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

# 🚨 启动安全校验：防御性编程
if not TG_BOT_TOKEN:
    raise ValueError("🚨 致命错误：TG_BOT_TOKEN 未在 .env 文件中配置或环境变量缺失！请检查。")

# 确保类型为 str（通过类型断言）
assert TG_BOT_TOKEN is not None, "TG_BOT_TOKEN must be a string"

# 🛡️ 授权用户白名单（从 .env 读取）
ALLOWED_TG_USERS = os.getenv("ALLOWED_TG_USERS", "")
ALLOWED_USER_IDS: list[int] = [
    int(user_id.strip()) 
    for user_id in ALLOWED_TG_USERS.split(",") 
    if user_id.strip().isdigit()
]

# 开启日志，方便在终端里看请求状态
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

async def keep_typing_action(chat_id: int, context):
    """🌟 无限幽灵输入：守护协程
    
    每隔 4 秒发送一次 typing 状态，直到被强行 Kill 掉
    
    Args:
        chat_id: Telegram 聊天 ID
        context: Telegram Context 对象
    """
    try:
        while True:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass  # 网络抖动时静默跳过，不中断心跳循环
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def _is_authorized_user(user_id: int) -> bool:
    """验证用户是否在授权白名单中
    
    Args:
        user_id: Telegram 用户唯一标识符
        
    Returns:
        bool: True 表示授权用户，False 表示未授权
    """
    # 未配置白名单时默认放行（向后兼容）
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def authorized(func):
    """
    🛡️ 权限校验装饰器：自动验证用户是否在授权白名单中
    
    用法：
        @authorized
        async def my_command(update, context):
            ...
    """
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        message = update.message
        
        # 验证有效性
        if user is None or message is None:
            logger.warning(f"收到无效的 {func.__name__} 请求（缺少用户或消息信息）")
            return
            
        # 权限校验
        if not _is_authorized_user(user.id):
            logger.warning(f"⛔ 未授权访问尝试 {func.__name__} | User ID: {user.id}")
            await message.reply_text("⛔ 未授权访问")
            return
        
        # 调用原函数
        return await func(update, context)
    
    return wrapper

async def send_dashboard(message_obj: Message, first_name: str):
    """
    下发全息操控面板 (解耦复用)
    
    Args:
        message_obj: Telegram Message 对象，用于回复
        first_name: 用户名字
    """
    keyboard = [
        [InlineKeyboardButton("💼 盘点当前持仓与盈亏", callback_data="cmd_portfolio")],
        [InlineKeyboardButton("📝 极速触发盘后研报", callback_data="cmd_daily_report")],
        [InlineKeyboardButton("🔔 设定盯盘价格预警", callback_data="cmd_alert")],
        [InlineKeyboardButton("📊 查询最新任务进度", callback_data="cmd_status")],
        [InlineKeyboardButton("📚 调阅历史情报档案", callback_data="cmd_kb_list")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (
        f"<blockquote><b>🚀 OmniStock 量化中台已挂载</b></blockquote>\n"
        f"指挥官 <b>{first_name}</b>，连接安全。\n\n"
        f"<i>您可以直接输入自然语言下达指令，或通过下方战术面板执行核心宏任务：</i>"
    )
    await message_obj.reply_text(welcome_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)


async def post_init(application: Application):
    """🤖 机器人启动时的钩子：强行改写客户端左下角的横线菜单"""
    await application.bot.set_my_commands([
        BotCommand("start", "🏠 唤醒主控台"),
        BotCommand("portfolio", "💼 盘点当前持仓与盈亏"),
        BotCommand("report", "📝 极速触发盘后研报"),
        BotCommand("status", "📊 查询最新任务进度"),
        BotCommand("kb", "📚 调阅历史情报档案"),
        BotCommand("alert", "🔔 设定盯盘价格预警"),
    ])
    logger.info("✅ 左下角全局菜单 (Bot Commands) 注入成功！")


@authorized
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    处理 /start 命令，调用面板生成器
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    user = update.effective_user
    message = update.message
    
    await send_dashboard(message, user.first_name)


@authorized
async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    快捷路由：个人持仓与盈亏盘点（调用大模型）
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    message = update.message
    
    # 调用大模型处理持仓核算
    user_msg = "帮我精确计算当前总市值和持仓盈亏，并生成财务明细报表。"
    await message.reply_text(
        f"<blockquote><b>⚡ 原生菜单指令注入：</b>\n<i>精确核算总市值</i></blockquote>",
        parse_mode=ParseMode.HTML
    )
    await execute_agent_task(user_msg, message, update.effective_user.id, context, update)


async def _dispatch_report_task(message: Message, job_id: str) -> None:
    """
    公共函数：派发研报任务到后台独立进程（复用逻辑）
    
    Args:
        message: Telegram Message 对象，用于发送回复
        job_id: 任务唯一 ID
    """
    # 1. 强行初始化本地状态文件
    status_dir = Path("./jobs/status").resolve()
    status_dir.mkdir(parents=True, exist_ok=True)
    initial_status = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    with open(status_dir / f"{job_id}.json", 'w', encoding='utf-8') as f:
        json.dump(initial_status, f, ensure_ascii=False, indent=2)
        
    # 2. 暴力弹射：直接唤醒 spawn_job.py 独立进程
    subprocess.Popen(
        [sys.executable, "spawn_job.py", "--job-id", job_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    
    # 3. 毫秒级 UI 反馈：直接组装带有"刷新进度"功能的面板
    reply_text = (
        f"✅ 研报任务已成功挂载至后台独立进程！\n\n"
        f"**任务 ID**: <code>{job_id}</code>\n\n"
        f"您可以继续聊天或锁屏手机。大约 3~4 分钟后，研报将自动推送到您的屏幕。"
    )
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 实时刷新任务进度", callback_data=f"check_job:{job_id}")]
    ])
    
    # 发送提示卡片
    await message.reply_text(reply_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


@authorized
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    快捷路由：极速触发盘后研报（脊髓反射模式，绕过大模型）
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    message = update.message
    
    try:
        # 1. 生成唯一任务 ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # 2. 调用公共派发函数
        await _dispatch_report_task(message, job_id)
        return
        
    except Exception as e:
        logger.error(f"/report 命令执行失败：{e}")
        await message.reply_text(f"⚠️ 任务派发失败：{type(e).__name__} - {str(e)}")


@authorized
async def kb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    快捷路由：调阅历史情报档案（调用大模型）
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    message = update.message
    
    # 调用大模型处理知识库查询
    user_msg = "列出知识库里现在有哪些文件可以读取？"
    await message.reply_text(
        f"<blockquote><b>⚡ 原生菜单指令注入：</b>\n<i>查看知识库文件</i></blockquote>",
        parse_mode=ParseMode.HTML
    )
    await execute_agent_task(user_msg, message, update.effective_user.id, context, update)


@authorized
async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    快捷路由：设定盯盘价格预警（引导用户）
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    message = update.message
    
    text = (
        "<blockquote><b>🔔 智能盯盘预警系统</b></blockquote>\n"
        "您现在可以使用自然语言下发盯盘指令，AI 会自动为您解析参数并挂载到后台监控引擎。\n\n"
        "<b>您可以直接这样对我说：</b>\n"
        "🗣️ <i>「帮我盯着英伟达，如果跌破 100 块叫我一声。」</i>\n"
        "🗣️ <i>「如果腾讯涨破了 350 港币，发消息提醒我减仓。」</i>"
    )
    await message.reply_text(text, parse_mode=ParseMode.HTML)

class AsyncTelegramCallbackHandler(AsyncCallbackHandler):
    """拦截 Agent 的异步执行流，实时动态更新到 Telegram 屏幕上"""
    
    def __init__(self, status_message: Message):
        self.status_message = status_message
        self.last_update_time = 0.0
        self.written_md_files: set[str] = set()

    async def _safe_edit(self, text: str):
        """防抖更新，避免触发 Telegram 严格的 API 频率限制 (Rate Limit)"""
        current_time = time.time()
        # 限制更新频率为最高 1 秒/次
        if current_time - self.last_update_time < 1.0:
            await asyncio.sleep(1.0 - (current_time - self.last_update_time))
            
        try:
            await self.status_message.edit_text(text, parse_mode=ParseMode.HTML)
            self.last_update_time = time.time()
        except Exception:
            pass  # 忽略 "Message is not modified" 等冗余报错

    async def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """当大模型决定调用某个工具时触发"""
        tool_name = serialized.get("name", "tool")

        if tool_name == "write_local_file":
            try:
                params = ast.literal_eval(input_str)
                file_path = params.get("file_path", "")
                if file_path.endswith(".md"):
                    self.written_md_files.add(Path(file_path).name)
            except (ValueError, SyntaxError, AttributeError):
                pass
        
        # 极客风的工具状态映射表
        tool_map = {
            # 行情查询类
            "get_universal_stock_price": "📈 正在拉取全球实时盘面数据...",
            "get_etf_price": "📊 正在拉取 ETF 基金核心数据...",
            # 绘图引擎类
            "draw_universal_stock_chart": "🎨 正在启动绘图引擎渲染 K 线...",
            # 搜索类
            "search_company_ticker": "🔍 正在全网检索股票代码...",
            # 文件操作类
            "read_local_file": "📂 正在穿透沙箱读取本地文件...",
            "write_local_file": "📝 正在排版并生成最终深度报告...",
            "list_kb_files": "🗂️ 正在扫描知识库文件索引...",
            # RAG 检索类
            "analyze_local_document": "📚 正在穿透本地向量库检索研报...",
            # 记忆系统类
            "update_user_memory": "🧠 正在将关键信息写入长期记忆库...",
            "append_transaction_log": "📜 正在追加交易日志流水账...",
            # 财务计算类
            "calculate_exact_portfolio_value": "🧮 正在使用程序精确核算财务数据...",
            # 研报任务类
            "trigger_daily_report": "🚀 正在将研报任务投递至独立进程...",
            "query_job_status": "📡 正在追踪后台任务执行状态..."
        }
        msg = tool_map.get(tool_name, f"⚡ 正在挂载系统组件：{tool_name}...")
        
        ui_text = f"<blockquote><b>🤖 OmniStock 引擎运转中...</b></blockquote>\n<i>{msg}</i>"
        await self._safe_edit(ui_text)
        
    async def on_tool_end(self, output: str, **kwargs):
        """当工具执行完毕并返回数据给大模型时触发"""
        ui_text = f"<blockquote><b>🤖 OmniStock 引擎运转中...</b></blockquote>\n<i>✔️ 数据提取完毕，正在进行逻辑推理...</i>"
        await self._safe_edit(ui_text)


async def render_markdown_table_to_image(text: str) -> tuple[str, list[str]]:
    """
    🚀 终极视觉拦截器：利用 Playwright 浏览器内核，将 Markdown 表格渲染为具有 Bloomberg 质感的 Web UI 并精准截图。
    """
    # 放宽对分割线空格的容忍度，完美适配 | :--- | 格式
    table_pattern = re.compile(r'((?:\|.*\|\n)+[ \t]*\|?[ \t]*[-:]+[-| :]*\|?\n(?:\|.*\|\n?)+)')
    matches = table_pattern.findall(text)
    
    if not matches:
        return text, []

    image_paths = []
    
    # 👑 极客级 CSS：暗黑金融终端质感
    css = """
    :root { --bg: #1A1D21; --border: #2D3239; --text: #E3E5E8; --header-bg: #22262B; --stripe: #1E2126; }
    html, body {
        background-color: var(--bg);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        margin: 0; padding: 20px;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    #capture_area {
        /* 🌟 核心修复 1：摒弃 inline-block，使用 max-content 绝对贴合内容 */
        width: max-content; 
        background-color: var(--bg); 
        padding: 16px;
        border: 1px solid var(--border);
        /* 🌟 核心修复 2：更改盒模型计算方式，防止 padding 引起的亚像素抖动 */
        box-sizing: border-box;
        /* 🌟 核心修复 3：暴力裁切，把超出整数边界的 0.x 像素底色直接切掉 */
        overflow: hidden;
    }
    table { border-collapse: collapse; color: var(--text); font-size: 14px; margin: 0; }
    th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }
    th {
        background-color: var(--header-bg); font-weight: 600; color: #A0A5AD;
        text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px;
    }
    tr:last-child td { border-bottom: none; }
    tr:nth-child(even) td { background-color: var(--stripe); }
    """
    
    # 启动极其轻量的无头浏览器
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu'])
        # 🌟 核心高分屏魔法：开启 3 倍设备像素比 (相当于苹果 Retina 视网膜屏物理超采样)
        page = await browser.new_page(device_scale_factor=3)
        
        for idx, md_table in enumerate(matches):
            try:
                # 1. 纯净转换：Markdown 表格 -> HTML <table>
                html_table = markdown.markdown(md_table, extensions=['tables'])
                
                # 2. 组装完整网页
                full_html = f"<!DOCTYPE html><html><head><style>{css}</style></head><body><div id='capture_area'>{html_table}</div></body></html>"
                
                # 3. 注入浏览器并等待渲染
                await page.set_content(full_html)
                
                # 4. 魔法时刻：精准锁定 div，无视背景进行像素级裁切截图！
                img_filename = f"table_render_{int(time.time())}_{idx}.png"
                img_path = (SANDBOX_DIR / img_filename).resolve()
                
                element = await page.wait_for_selector('#capture_area')
                
                # 🌟 终极防线：注入 JS 计算真实亚像素宽度，并向上取整锁定物理像素，彻底封死右侧缝隙！
                await element.evaluate("el => el.style.width = Math.ceil(el.getBoundingClientRect().width) + 'px'")
                
                await element.screenshot(path=str(img_path), omit_background=True)
                
                image_paths.append(str(img_path))
                
                # 5. 替换原文文本
                img_markdown = f"\n\n![表格](./{img_filename})\n\n"
                text = text.replace(md_table, img_markdown)
                
            except Exception as e:
                logger.error(f"Playwright 表格渲染失败：{e}")
                continue
                
        await browser.close()
            
    return text, image_paths


async def convert_md_to_pdf(md_path: Path) -> Path:
    """
    将 agent_workspace 中的 .md 文件渲染为 PDF，内嵌图片。

    Args:
        md_path: .md 文件的绝对路径（必须在 SANDBOX_DIR 内）

    Returns:
        Path: 生成的 .pdf 文件路径（与 .md 同目录）

    Raises:
        Exception: Playwright 渲染失败时向上抛出
    """
    md_text = md_path.read_text(encoding='utf-8')

    # MD -> HTML（启用表格 + 代码块扩展）
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # 将 <img src="./xxx.png"> 中的相对路径替换为 file:// 绝对路径，确保 Playwright 能加载图片
    def resolve_img_src(match: re.Match) -> str:
        src = match.group(1)
        img_file = (SANDBOX_DIR / Path(src).name).resolve()
        if img_file.exists():
            return f'src="file://{img_file}"'
        return match.group(0)

    html_body = re.sub(r'src="([^"]+)"', resolve_img_src, html_body)

    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
                     "Microsoft YaHei", "Segoe UI", Roboto, Arial, sans-serif;
        font-size: 14px; line-height: 1.8; color: #222;
        max-width: 860px; margin: 0 auto; padding: 0 8px;
    }
    h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 6px; }
    h2 { font-size: 18px; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 24px; }
    h3 { font-size: 15px; margin-top: 18px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
    th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
    th { background: #f0f0f0; font-weight: 600; }
    tr:nth-child(even) td { background: #fafafa; }
    img { max-width: 100%; height: auto; display: block; margin: 12px auto; }
    code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 12px; }
    pre { background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }
    pre code { background: none; padding: 0; }
    hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
    blockquote { border-left: 3px solid #aaa; margin: 0; padding-left: 12px; color: #555; }
    """

    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{css}</style>
</head><body>{html_body}</body></html>"""

    pdf_path = md_path.with_suffix('.pdf')
    tmp_html = md_path.with_suffix('.tmp.html')

    try:
        tmp_html.write_text(full_html, encoding='utf-8')
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu'])
            page = await browser.new_page()
            await page.goto(f'file://{tmp_html}', wait_until='networkidle')
            await page.pdf(
                path=str(pdf_path),
                format='A4',
                margin={'top': '18mm', 'bottom': '18mm', 'left': '15mm', 'right': '15mm'},
                print_background=True
            )
            await browser.close()
    finally:
        tmp_html.unlink(missing_ok=True)

    return pdf_path


def translate_to_telegram_html(text: str) -> str:
    """将标准 Markdown 安全地转换为 Telegram HTML 方言，提升 UI 质感"""
    if not text:
        return text
    
    # 1. 基础转义 (防御性编程：防止 LLM 吐出的 < > 破坏 HTML 结构)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # 2. 渲染代码块与行内代码
    text = re.sub(
        r'```(\w+)?\n(.*?)```', 
        lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>', 
        text, flags=re.DOTALL
    )
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # 3. 渲染加粗与删除线
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
    
    # 4. 🌟 核心 UI 升级：标题降级为带前缀的引用块 <blockquote>
    # 拦截 4 级及以上标题（终极兜底，渲染为加粗子菱形，不再嵌套 blockquote）
    text = re.sub(r'^#{4,}\s+(.*)', r'<b>◈ \1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*)', r'<blockquote><b>■ \1</b></blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*)', r'<blockquote><b>● \1</b></blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*)', r'<blockquote><b>◆ \1</b></blockquote>', text, flags=re.MULTILINE)
    
    # 5. 渲染超链接
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    
    # 6. 🌟 列表语法降级与美化 (Telegram HTML 不支持 ul/li)
    # 处理二级缩进列表 (匹配 1 个以上空格或制表符开头的 - 或 *) -> 转换为终端树状节点
    text = re.sub(r'^([ \t]{1,})[-*]\s+', r'\1└─ ▫️ ', text, flags=re.MULTILINE)
    # 处理一级列表 (顶格的 - 或 *) -> 转换为醒目的蓝色主节点
    text = re.sub(r'^[-*]\s+', r'🔹 ', text, flags=re.MULTILINE)
    
    # 7. 🌟 静默擦除水平分割线 (Telegram 不支持 <hr> 标签)
    # 匹配单独一行的 ---, ***, 或 ___ 并直接删除
    text = re.sub(r'^[-*_]{3,}\s*\n?', '', text, flags=re.MULTILINE)
    
    return text


async def send_with_caption_split(
    message,
    photo,
    caption: str,
    max_length: int = 1024
):
    """
    发送图片，如果 caption 超长则拆分成多条消息。
    
    Args:
        message: Telegram Message 对象
        photo: 图片文件对象
        caption: 完整 caption 文本
        max_length: Telegram caption 最大长度（1024 字符）
    """
    if len(caption) <= max_length:
        # 正常发送
        try:
            await message.reply_photo(
                photo=photo,
                caption=caption,
                parse_mode=ParseMode.HTML,
                show_caption_above_media=True
            )
        except Exception as e:
            logger.warning(f"Caption HTML 渲染失败，降级为纯文本：{e}")
            fallback_caption = re.sub(r'<[^>]+>', '', caption)
            await message.reply_photo(photo=photo, caption=fallback_caption, show_caption_above_media=True)
    else:
        # Caption 超长拆分：
        # 1. 第一条：图片 + caption 前段（截断）
        # 2. 后续：纯文本气泡（剩余部分）
        
        caption_part1 = caption[:max_length-3] + "..."
        try:
            await message.reply_photo(
                photo=photo,
                caption=caption_part1,
                parse_mode=ParseMode.HTML,
                show_caption_above_media=True
            )
        except Exception as e:
            logger.warning(f"Caption HTML 渲染失败，降级为纯文本：{e}")
            fallback_caption = re.sub(r'<[^>]+>', '', caption_part1)
            await message.reply_photo(photo=photo, caption=fallback_caption, show_caption_above_media=True)
        
        await asyncio.sleep(0.2)
        
        # 剩余部分作为纯文本发送（修复吞字 Bug：从截断处无缝衔接）
        remaining = caption[max_length-3:]
        while remaining:
            if len(remaining) > 4096:  # Telegram 文本消息上限
                text_chunk = remaining[:4096]
                remaining = remaining[4096:]
            else:
                text_chunk = remaining
                remaining = ""
            
            try:
                await message.reply_text(text_chunk, parse_mode=ParseMode.HTML)
            except Exception as e:
                fallback = re.sub(r'<[^>]+>', '', text_chunk)
                await message.reply_text(fallback)
            
            await asyncio.sleep(0.2)


async def execute_agent_task(
    user_msg: str, 
    message: Message, 
    user_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    update: Update
):
    """
    抽象后的核心任务执行流，可被文本消息或按钮点击复用
    
    Args:
        user_msg: 用户输入的原始消息或按钮映射的 Prompt
        message: Telegram Message 对象，用于发送回复
        user_id: 用户唯一标识符，用于 session 隔离
        context: Telegram Context 对象，包含 bot 实例和会话状态
        update: Telegram Update 对象，用于获取 chat 信息
    """
    logger.info(f"[execute_agent_task] 开始执行任务 | user_id={user_id}, msg_length={len(user_msg)}")
    
    effective_chat = update.effective_chat
    if effective_chat is None:
        logger.warning("无法获取 effective_chat，终止任务")
        return
    
    chat_id = effective_chat.id
    
    # 🌟 1. 启动无限心跳引擎 (后台运行，不阻塞主线程)
    typing_task = asyncio.create_task(keep_typing_action(chat_id, context))
    
    status_msg: Message | None = None
    
    try:
        # 🌟 2. 下发初始高颜值占位符
        status_msg = await message.reply_text(
            "<blockquote><b>🤖 OmniStock 引擎已唤醒</b></blockquote>\n<i>⏳ 正在建立神经连接...</i>", 
            parse_mode=ParseMode.HTML
        )
        
        # 🌟 3. 实例化你的专属 Telegram 回调拦截器
        tg_callback = AsyncTelegramCallbackHandler(status_msg)
        # 🚀 3. 异步唤醒底层 AI 引擎，并将拦截器强行注入配置 (Config)！
        response = await agent_with_chat_history.ainvoke(
            {
                "input": user_msg,
                "user_profile": get_user_profile(),
                "current_time": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
            },
            config={
                "configurable": {"session_id": f"tg_session_{user_id}"},
                "callbacks": [tg_callback]  # 👈 核心挂载点
            }
        )
        
        reply_text = response['output']
        
        # 🌟 4. 推理结束后，依然保留删掉占位符的逻辑，保持界面整洁
        await status_msg.delete()
        
        # 1. 🎯 触发表格视觉拦截器（表格图片已转为 Markdown 语法插入原文本）
        final_text, _ = await render_markdown_table_to_image(reply_text)
        
        # ==========================================
        # 🚀 2. 终极渲染引擎：图片携带前置文本作为 caption
        # ==========================================
        # 使用正则表达式，将文本按照 ![描述](路径) 切片，保留图片语法本身作为独立块
        chunks = re.split(r'(!\[.*?\]\(.*?\))', final_text)
        
        # 消费标记数组：记录哪些文本切片已作为 caption 发送
        is_consumed = [False] * len(chunks)
        
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # 判断当前切片是不是图片标记
            img_match = re.match(r'^!\[.*?\]\((.*?)\)$', chunk)
            
            if img_match:
                # 🖼️ 图片切片，走图片发送管道
                img_filename = img_match.group(1).replace("./", "")
                img_path = (SANDBOX_DIR / img_filename).resolve()
                
                # 读取前一个文本切片作为 caption（如果前一个是文本且未消费）
                raw_caption = ""
                if i > 0 and not is_consumed[i-1]:
                    prev_chunk = chunks[i-1].strip()
                    # 检查前一个切片是否是文本（不是图片）
                    prev_is_image = re.match(r'^!\[.*?\]\(.*?\)$', prev_chunk)
                    if not prev_is_image:
                        raw_caption = prev_chunk
                
                if img_path.exists():
                    try:
                        with open(img_path, 'rb') as photo:
                            if raw_caption:
                                html_caption = translate_to_telegram_html(raw_caption)
                                await send_with_caption_split(
                                    message, photo, html_caption
                                )
                                is_consumed[i-1] = True
                            else:
                                # 纯图片发送（无前置文本）
                                await message.reply_photo(photo=photo)
                    except Exception as e:
                        logger.error(f"发送图片失败：{e}")
                else:
                    await message.reply_text("⚠️ [此处图片生成失败或已被清理]")
                
                # 微小延迟，保证 Telegram 服务器按顺序排布气泡
                await asyncio.sleep(0.2)
                
            else:
                # 📝 纯文本切片，走文本发送管道
                
                # 预读：检查下一个切片是否是图片
                next_is_image = (
                    i + 1 < len(chunks) and
                    re.match(r'^!\[.*?\]\(.*?\)$', chunks[i+1].strip())
                )
                
                if next_is_image:
                    # 跳过发送，等图片发送时作为 caption 一起发
                    continue
                else:
                    # HTML 渲染后发送文本（带防御性降级）
                    html_text = translate_to_telegram_html(chunk)
                    try:
                        await message.reply_text(
                            html_text, parse_mode=ParseMode.HTML
                        )
                    except Exception as e:
                        logger.warning(f"HTML 渲染失败，降级为纯文本发送：{e}")
                        fallback_text = re.sub(r'<[^>]+>', '', html_text)
                        await message.reply_text(fallback_text)
                    
                    # 微小延迟，锁死文章阅读流顺序
                    await asyncio.sleep(0.2)
        
        # 🌟 5. 若本次写入了 .md 文件，展示 Inline Keyboard 供用户按需下载
        if tg_callback.written_md_files:
            buttons = [
                [InlineKeyboardButton(f"📄 {md_name}", callback_data=f"send_file:{md_name}")]
                for md_name in tg_callback.written_md_files
            ]
            await message.reply_text(
                "📁 以下报告文件已生成，点击即可获取：",
                reply_markup=InlineKeyboardMarkup(buttons)
            )

    except Exception as e:
        logger.error(f"[execute_agent_task] 处理失败：{e}")
        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass
        await message.reply_text(f"⚠️ 系统熔断：{str(e)}")
    finally:
        # 🛑 无论大模型成功还是崩溃报错，必须在 finally 里精准射杀心跳协程！
        typing_task.cancel()
        try:
            await typing_task
        except (asyncio.CancelledError, Exception):
            pass


async def price_watcher_routine(context: ContextTypes.DEFAULT_TYPE):
    """🌟 纯 Python 轻量级盯盘引擎 (每 5 分钟执行，0 Token 消耗)"""
    import logging
    
    alerts_file = Path("./memory/alerts.json").resolve()
    if not alerts_file.exists():
        return
        
    try:
        with open(alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
    except Exception:
        return
        
    if not alerts:
        return
        
    changed = False
    
    for chat_id_str, user_tasks in list(alerts.items()):
        chat_id = int(chat_id_str)
        triggered_keys = []
        
        for task_key, task_info in user_tasks.items():
            ticker = task_info['ticker']
            operator = task_info['operator']
            target_price = float(task_info['target_price'])
            
            try:
                price_data = fetch_stock_price_raw(ticker)
                current_price = float(price_data.get('close', 0))
                if current_price == 0:
                    continue
                
                is_triggered = False
                if operator == '<' and current_price <= target_price:
                    is_triggered = True
                elif operator == '>' and current_price >= target_price:
                    is_triggered = True
                    
                if is_triggered:
                    alert_msg = (
                        f"<blockquote><b>🚨 智能盯盘触发警告</b></blockquote>\n"
                        f"标的代码：<b>{ticker}</b>\n"
                        f"预警条件：{operator} {target_price}\n"
                        f"当前最新价：<b>{current_price}</b>\n\n"
                        f"<i>系统已自动将此单次预警任务销毁。</i>"
                    )
                    await context.bot.send_message(chat_id=chat_id, text=alert_msg, parse_mode=ParseMode.HTML)
                    triggered_keys.append(task_key)
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"预警查价失败 {ticker}: {e}")
                
        for key in triggered_keys:
            del alerts[chat_id_str][key]
            changed = True
            
    if changed:
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)


def _read_job_status_sync(job_id: str) -> str:
    """⚡ 脊髓反射：直接读取本地 JSON，零延迟返回，绝对不调用大模型"""
    try:
        status_file = Path(f"./jobs/status/{job_id}.json").resolve()
        if not status_file.exists():
            return f"❌ 未找到任务 <code>{job_id}</code> 的状态文件。"
        
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
            
        status_map = {
            "pending": "⏳ 等待分配资源...",
            "running": "🔄 华尔街虚拟交易室正在激烈辩论与计算中...",
            "completed": "✅ 研报已生成并推送到您的屏幕！",
            "failed": "❌ 任务执行崩溃"
        }
        
        current_status = status_map.get(status.get('status'), '未知状态')
        
        result = (
            f"<blockquote><b>📊 任务雷达跟踪：{job_id}</b></blockquote>\n"
            f"<b>当前状态：</b>{current_status}\n"
            f"<b>创建时间：</b><code>{status.get('created_at', 'N/A')}</code>\n"
        )
        if status.get('started_at'):
            result += f"<b>启动时间：</b><code>{status.get('started_at')}</code>\n"
        if status.get('completed_at'):
            result += f"<b>完成时间：</b><code>{status.get('completed_at')}</code>\n"
        if status.get('error'):
            result += f"<b>异常抛出：</b><code>{status.get('error')}</code>\n"
            
        return result
    except Exception as e:
        return f"❌ 状态读取物理层异常：{e}"


def _get_latest_job_id() -> str:
    """⚡ 硬盘级嗅探：扫描本地状态目录，获取最新提交的任务 ID"""
    status_dir = Path("./jobs/status").resolve()
    if not status_dir.exists():
        return ""
    
    files = list(status_dir.glob("*.json"))
    if not files:
        return ""
    
    # 按照文件的最后修改时间降序排序，提取最新的那个文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file.stem  # 返回去掉后缀的纯 job_id


async def _handle_status_query(message: Message) -> None:
    """
    内部函数：处理状态查询逻辑（复用核心逻辑）
    
    Args:
        message: Telegram Message 对象
    """
    latest_job_id = _get_latest_job_id()
    
    if not latest_job_id:
        await message.reply_text("📭 当前系统没有任何后台任务记录。")
        return
        
    status_text = _read_job_status_sync(latest_job_id)
    
    refresh_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 实时刷新任务进度", callback_data="check_job:latest")]
    ])
    await message.reply_text(status_text, parse_mode=ParseMode.HTML, reply_markup=refresh_keyboard)


@authorized
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """处理来自左下角菜单的 /status 快捷指令"""
    message = update.message
    if message is None:
        return
    await _handle_status_query(message)


@authorized
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    文本消息处理中枢
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    user = update.effective_user
    message = update.message
    user_msg = message.text or ""
    
    logger.info(f"收到用户 {user.id} 的消息：{user_msg}")
    
    # 将任务丢给核心执行流
    await execute_agent_task(user_msg, message, user.id, context, update)


@authorized
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    🗂️ 移动端知识投喂中枢：拦截用户上传的文件，存入知识库并触发向量解析 (增强防御与计时版)
    """
    user = update.effective_user
    message = update.message
    doc = message.document
    
    file_name = doc.file_name or f"upload_{int(time.time())}.txt"
    file_size_mb = doc.file_size / (1024 * 1024) if doc.file_size else 0

    # 🚨 1. 物理防线：Telegram 标准 Bot API 硬性限制下载 20MB 以内的文件
    if file_size_mb > 20.0:
        await message.reply_text(f"⚠️ 系统熔断：文件高达 {file_size_mb:.1f}MB。受限于 Telegram 官方网关物理限制，系统最多只能接收 20MB 以内的研报。请压缩后重试。")
        return

    # 2. 格式拦截
    allowed_exts = {'.pdf', '.md', '.txt', '.csv'}
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in allowed_exts:
        await message.reply_text(f"⚠️ 格式拒绝：暂不支持 {ext} 格式进行向量化。")
        return

    # 3. 初始反馈：缓解下载焦虑
    status_msg = await message.reply_text(
        f"⏳ <b>正在从 Telegram 节点拉取文件：</b> {file_name} ({file_size_mb:.1f}MB)\n<blockquote><i>受限于跨国节点带宽，下载可能需要数秒至十几秒...</i></blockquote>", 
        parse_mode=ParseMode.HTML
    )

    try:
        start_dl_time = time.time()
        
        tg_file = await context.bot.get_file(doc.file_id)
        save_path = (KB_DIR / file_name).resolve()
        
        if not save_path.is_relative_to(KB_DIR):
            await status_msg.edit_text("❌ 安全拦截：非法的文件名，已销毁。")
            return

        # 物理下载
        await tg_file.download_to_drive(custom_path=save_path)
        dl_cost = time.time() - start_dl_time

        # 4. 🌟 UX 状态瞬间跳变：明确告知用户下载已完成，现在是算力消耗时间！
        await status_msg.edit_text(
            f"<blockquote><b>⚡ 物理传输完毕 (耗时 {dl_cost:.1f}s)</b></blockquote>\n"
            f"<i>已成功挂载至知识库：{file_name}。\n"
            f"🧠 正在唤醒阿里云 Embedding 引擎进行高维张量切片与 RAG 深度阅读，请稍候...</i>", 
            parse_mode=ParseMode.HTML
        )

        rag_prompt = f"我已经把一份名为 '{file_name}' 的文件放进了知识库。请调用 analyze_local_document 工具，仔细阅读这篇文档，并给我一份结构化的核心内容摘要。"
        
        # 将提示消息传给后续流以便任务完成后删除
        await execute_agent_task(rag_prompt, message, user.id, context, update)
        
        # 任务完毕后清理掉这条冗长的进度消息
        try:
            await status_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"文件接收与解析失败：{e}")
        await status_msg.edit_text(f"❌ 链路崩塌：文件处理失败 ({type(e).__name__})")


async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    拦截 Inline 按钮的点击事件，转化为 Agent Prompt
    
    Args:
        update: Telegram Update 对象
        context: Telegram Context 对象
    """
    query = update.callback_query
    if query is None:
        logger.warning("收到无效的回调查询")
        return
        
    await query.answer()  # 必须应答，否则按钮会一直转圈
    
    user_id = query.from_user.id if query.from_user else None
    if user_id is None or not _is_authorized_user(user_id):
        logger.warning(f"⛔ 未授权按钮点击 | User ID: {user_id}")
        return
    
    cmd = query.data
    if cmd is None:
        logger.warning("按钮指令数据为空")
        return
    
    # 🌟 1. 状态物理销毁：大部分按钮点击后立刻清除防止重放。但刷新按钮需要保留！
    if not cmd.startswith("check_job:"):
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as e:
            logger.warning(f"清除按钮失败：{e}")

    # 🌟 2. 旁路直通网关 (Spinal Reflex)：拦截刷新指令，0.1 秒极速响应，绕过大模型！
    if cmd.startswith("check_job:"):
        job_id = cmd.split(":")[1]
        
        # 如果是 latest 标记，每次点击都重新获取最新 job_id
        if job_id == "latest":
            job_id = _get_latest_job_id()
            if not job_id:
                await query.message.edit_text("📭 当前系统没有任何后台任务记录。", parse_mode=ParseMode.HTML)
                return
        
        status_text = _read_job_status_sync(job_id)
        
        # 重新生成带刷新功能的面板（只保留刷新按钮）
        refresh_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 实时刷新任务进度", callback_data=cmd)]
        ])
        
        try:
            await query.message.edit_text(status_text, parse_mode=ParseMode.HTML, reply_markup=refresh_keyboard)
        except Exception:
            pass # 忽略 Telegram "内容未发生改变 (Message is not modified)" 的冗余报错
        return

    # 🌟 3. 文件按需下载：将 .md 转为 PDF 后发送
    if cmd.startswith("send_file:"):
        md_name = cmd.split(":", 1)[1]
        md_path = (SANDBOX_DIR / md_name).resolve()
        if not md_path.exists():
            await query.message.reply_text(f"⚠️ 文件不存在或已被清理：{md_name}")
            return
        try:
            await query.message.reply_text("⏳ 正在渲染 PDF，请稍候...")
            pdf_path = await convert_md_to_pdf(md_path)
            pdf_name = pdf_path.name
            with open(pdf_path, 'rb') as doc:
                await query.message.reply_document(
                    document=doc,
                    filename=pdf_name,
                    caption=f"📑 <b>{pdf_name}</b>",
                    parse_mode=ParseMode.HTML
                )
        except Exception as e:
            logger.error(f"PDF 转换或发送失败 [{md_name}]：{e}")
            await query.message.reply_text(f"⚠️ PDF 生成失败：{e}")
        return

    # 🌟 4. 闭环路由：拦截返回主控台的指令
    if cmd == "cmd_home":
        try:
            if query.message and isinstance(query.message, Message):
                await query.message.delete()
        except Exception:
            pass
        if query.message and isinstance(query.message, Message):
            await send_dashboard(query.message, query.from_user.first_name)
        return

    user_msg = ""
    
    # 路由表：将隐藏指令映射为精确的工程 Prompt
    if cmd == "cmd_portfolio":
        user_msg = "帮我精确计算当前总市值和持仓盈亏，并生成财务明细报表。"
        
    elif cmd == "cmd_daily_report":
        # ⚡ 脊髓反射启动：复用公共派发函数
        
        # 1. 生成唯一任务 ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # 2. 调用公共派发函数（需要 query.message 转为 Message 对象）
        if query.message and isinstance(query.message, Message):
            await _dispatch_report_task(query.message, job_id)
        return
        
    elif cmd == "cmd_status":
        # 🌟 脊髓反射：复用内部状态查询逻辑（权限已在函数入口处验证）
        if query.message and isinstance(query.message, Message):
            await _handle_status_query(query.message)
        return
        
    elif cmd == "cmd_kb_list":
        user_msg = "列出知识库里现在有哪些文件可以读取？"
        
    elif cmd == "cmd_alert":
        text = (
            "<blockquote><b>🔔 智能盯盘预警系统</b></blockquote>\n"
            "您现在可以使用自然语言下发盯盘指令，AI 会自动为您解析参数并挂载到后台监控引擎。\n\n"
            "<b>您可以直接这样对我说：</b>\n"
            "🗣️ <i>「帮我盯着英伟达，如果跌破 100 块叫我一声。」</i>\n"
            "🗣️ <i>「如果腾讯涨破了 350 港币，发消息提醒我减仓。」</i>"
        )
        if query.message:
            await query.message.reply_text(text, parse_mode=ParseMode.HTML)
        return
        
    else:
        logger.warning(f"未知按钮指令：{cmd}")
        return
        
    if user_msg and query.message:
        # 在界面上回显"幽灵输入"，提升极客感（永久保留作为历史轨迹）
        await query.message.reply_text(  # type: ignore
            f"<blockquote><b>⚡ 战术面板指令注入：</b>\n<i>{user_msg}</i></blockquote>",
            parse_mode=ParseMode.HTML
        )
        # 复用核心执行流，无缝对接大模型
        await execute_agent_task(user_msg, query.message, user_id, context, update)  # type: ignore


async def broadcast_to_telegram(text: str):
    """
    🌟 服务端主动推送引擎：供 daily_job 跨进程调用，复用高级图文渲染引擎下发研报
    
    Args:
        text: 待发送的研报 Markdown 文本
    """
    if not TG_BOT_TOKEN or not ALLOWED_USER_IDS:
        logger.warning("未配置 Telegram Token 或白名单，无法进行推送。")
        return
        
    bot = Bot(token=TG_BOT_TOKEN)
    
    # 1. 拦截并使用 Playwright 渲染表格
    final_text, _ = await render_markdown_table_to_image(text)
    
    # 2. 图文切片混排
    chunks = re.split(r'(!\[.*?\]\(.*?\))', final_text)
    
    for user_id in ALLOWED_USER_IDS:
        try:
            # 播报报头
            await bot.send_message(
                chat_id=user_id, 
                text="<blockquote><b>🔔 OmniStock 每日宏观与账户研报已送达</b></blockquote>", 
                parse_mode=ParseMode.HTML
            )
            
            # 依次发送切片
            is_consumed = [False] * len(chunks)
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk: continue
                
                img_match = re.match(r'^!\[.*?\]\((.*?)\)$', chunk)
                if img_match:
                    img_filename = img_match.group(1).replace("./", "")
                    img_path = (SANDBOX_DIR / img_filename).resolve()
                    
                    raw_caption = ""
                    if i > 0 and not is_consumed[i-1]:
                        prev_chunk = chunks[i-1].strip()
                        if not re.match(r'^!\[.*?\]\(.*?\)$', prev_chunk):
                            raw_caption = prev_chunk
                            
                    if img_path.exists() and img_path.stat().st_size > 0:
                        with open(img_path, 'rb') as photo:
                            if raw_caption:
                                html_caption = translate_to_telegram_html(raw_caption)
                                # 降级截断处理
                                try:
                                    if len(html_caption) <= 1024:
                                        await bot.send_photo(chat_id=user_id, photo=photo, caption=html_caption, parse_mode=ParseMode.HTML, show_caption_above_media=True)
                                    else:
                                        # 开启图片沉底魔法
                                        await bot.send_photo(chat_id=user_id, photo=photo, caption=html_caption[:1021]+"...", parse_mode=ParseMode.HTML, show_caption_above_media=True)
                                        # 修复吞字 Bug
                                        await bot.send_message(chat_id=user_id, text=html_caption[1021:], parse_mode=ParseMode.HTML)
                                except Exception:
                                    fallback = re.sub(r'<[^>]+>', '', html_caption)
                                    # 降级模式也开启魔法参数
                                    await bot.send_photo(chat_id=user_id, photo=photo, caption=fallback[:1024], show_caption_above_media=True)
                                is_consumed[i-1] = True
                            else:
                                await bot.send_photo(chat_id=user_id, photo=photo)
                    else:
                        logger.warning(f"图片文件不存在或为空：{img_path}")
                    await asyncio.sleep(0.3) # 防封锁限流
                    
                else:
                    next_is_image = (i + 1 < len(chunks) and re.match(r'^!\[.*?\]\(.*?\)$', chunks[i+1].strip()))
                    if next_is_image: continue
                    
                    html_text = translate_to_telegram_html(chunk)
                    try:
                        await bot.send_message(chat_id=user_id, text=html_text, parse_mode=ParseMode.HTML)
                    except Exception:
                        fallback = re.sub(r'<[^>]+>', '', html_text)
                        await bot.send_message(chat_id=user_id, text=fallback)
                    await asyncio.sleep(0.3)
                    
        except Exception as e:
            logger.error(f"向用户 {user_id} 推送失败：{e}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """全局错误处理器：捕获所有未处理的异常"""
    logger.error("未捕获的异常：", exc_info=context.error)
    
    # 网络错误时静默处理，避免刷屏
    if isinstance(context.error, Exception):
        error_type = type(context.error).__name__
        if "ConnectError" in error_type or "NetworkError" in error_type:
            logger.warning("检测到网络错误，可能是临时 DNS 解析失败或 Telegram 服务器不可达")
            return


def main():
    """启动机器人"""
    logger.info("启动 OmniStock Telegram Bot...")
    
    # 👑 架构师指令：暴力破解网络超时限制，把等待时间拉满！
    application = (
        Application.builder()
        .token(str(TG_BOT_TOKEN))
        .read_timeout(120)       # 读取超时放宽到 120 秒
        .write_timeout(120)      # 写入（发送大图片）超时放宽到 120 秒
        .connect_timeout(60)     # 连接超时放宽到 60 秒
        .pool_timeout(120)       # 连接池超时放宽
        .post_init(post_init)    # 👈 核心：在此处挂载生命周期钩子
        .build()
    )
    
    # 注册全局错误处理器
    application.add_error_handler(error_handler)

    # 注册处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("kb", kb_command))
    application.add_handler(CommandHandler("alert", alert_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(CallbackQueryHandler(handle_button_click))

    # 🌟 挂载每 5 分钟一次的纯 Python 盯盘巡检器 (0 Token 消耗)
    if application.job_queue:
        application.job_queue.run_repeating(price_watcher_routine, interval=300, first=10)

    # 启动长轮询，Bot 会一直挂在后台监听
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()