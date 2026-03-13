import os
import re
import time
import logging
import asyncio
from datetime import datetime
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# 类型提示
from typing import List

from main import SANDBOX_DIR

# 🚀 新视觉引擎依赖
import markdown
from playwright.async_api import async_playwright

# 🌟 无缝引入咱们精心打磨的底层 Agent 引擎
from main import agent_with_chat_history, get_user_profile

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

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """当用户发送 /start 时触发的欢迎语"""
    user = update.effective_user
    message = update.message
    
    # 安全检查：确保 user 和 message 存在
    if user is None or message is None:
        logger.warning("收到无效的 /start 请求（缺少用户或消息信息）")
        return
        
    if not _is_authorized_user(user.id):
        logger.warning(f"⛔ 未授权访问尝试 /start 命令 | User ID: {user.id}")
        await message.reply_text("⛔ 未授权访问")
        return
    await message.reply_text(
        f"你好，{user.first_name}！我是 OmniStock Agent。\n随时向我发送股票代码或询问大盘分析。"
    )

async def render_markdown_table_to_image(text: str) -> tuple[str, list[str]]:
    """
    🚀 终极视觉拦截器：利用 Playwright 浏览器内核，将 Markdown 表格渲染为具有 Bloomberg 质感的 Web UI 并精准截图。
    """
    table_pattern = re.compile(r'((?:\|.*\|\n)+\|?(?:[-:]+[-| :]*)\|?\n(?:\|.*\|\n?)+)')
    matches = table_pattern.findall(text)
    
    if not matches:
        return text, []

    image_paths = []
    
    # 👑 极客级 CSS：暗黑金融终端质感
    css = """
    :root { --bg: #1A1D21; --border: #2D3239; --text: #E3E5E8; --header-bg: #22262B; --stripe: #1E2126; }
    body {
        background-color: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        margin: 0; padding: 20px;
    }
    #capture_area {
        display: inline-block; background-color: var(--bg); padding: 16px;
        border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 8px 24px rgba(0,0,0,0.4);
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
        page = await browser.new_page()
        
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
    text = re.sub(r'^###\s+(.*)', r'<blockquote><b>■ \1</b></blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*)', r'<blockquote><b>● \1</b></blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*)', r'<blockquote><b>◆ \1</b></blockquote>', text, flags=re.MULTILINE)
    
    # 5. 渲染超链接
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    
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
        
        # 剩余部分作为纯文本发送
        remaining = caption[max_length:]
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """核心消息处理中枢"""
    user = update.effective_user
    message = update.message
    
    # 安全检查：确保 user 和 message 存在
    if user is None or message is None:
        logger.warning("收到无效的消息请求（缺少用户或消息信息）")
        return
        
    user_id = user.id
    if not _is_authorized_user(user_id):
        message_text = message.text if message.text is not None else ""
        logger.warning(f"⛔ 未授权消息请求 | User ID: {user_id} | Message: {message_text[:50]}...")
        return  # 静默拒绝，不暴露任何系统信息
    user_msg = message.text or ""
    
    logger.info(f"收到用户 {user_id} 的消息: {user_msg}")
    
    # 🌟 缓兵之计：立马给手机发一条提示，并显示顶部的"正在输入..."状态
    status_msg = await update.message.reply_text("⏳ 正在拉取底层数据并深度推理，请稍候...")

    effective_chat = update.effective_chat
    if effective_chat is not None:
        await context.bot.send_chat_action(chat_id=effective_chat.id, action='typing')
    
    try:
        # 🚀 异步唤醒底层 AI 引擎
        response = await agent_with_chat_history.ainvoke(
            {
                "input": user_msg,
                "user_profile": get_user_profile(),
                "current_time": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
            },
            config={"configurable": {"session_id": f"tg_session_{user_id}"}}
        )
        
        reply_text = response['output']
        
        # 🌟 拿到结果后，先把"缓兵之计"的消息删掉，保持聊天界面整洁
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
                
                if img_path.exists() and update.message is not None:
                    try:
                        with open(img_path, 'rb') as photo:
                            if raw_caption:
                                html_caption = translate_to_telegram_html(raw_caption)
                                await send_with_caption_split(
                                    update.message, photo, html_caption
                                )
                                is_consumed[i-1] = True
                            else:
                                # 纯图片发送（无前置文本）
                                await update.message.reply_photo(photo=photo)
                    except Exception as e:
                        logger.error(f"发送图片失败：{e}")
                else:
                    if update.message is not None:
                        await update.message.reply_text("⚠️ [此处图片生成失败或已被清理]")
                
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
                    if update.message is not None:
                        html_text = translate_to_telegram_html(chunk)
                        try:
                            await update.message.reply_text(
                                html_text, parse_mode=ParseMode.HTML
                            )
                        except Exception as e:
                            logger.warning(f"HTML 渲染失败，降级为纯文本发送：{e}")
                            fallback_text = re.sub(r'<[^>]+>', '', html_text)
                            await update.message.reply_text(fallback_text)
                        
                        # 微小延迟，锁死文章阅读流顺序
                        await asyncio.sleep(0.2)
        
    except Exception as e:
        logger.error(f"处理失败：{e}")
        # 🌟 确保即使出错也要删除等待提示，避免残留
        await status_msg.delete()
        message = update.message
        if message is not None:
            await message.reply_text(f"⚠️ 系统熔断：{str(e)}")

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
        .build()
    )

    # 注册处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动长轮询，Bot 会一直挂在后台监听
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()