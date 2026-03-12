import os
import re
import time
import logging
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# 类型提示
from typing import List

# 🌟 绘图引擎依赖
import pandas as pd
import matplotlib.pyplot as plt
from main import SANDBOX_DIR

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

# 解决 matplotlib 中文显示方块字的问题 (使用容器内安装的文泉驿字体)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def render_markdown_table_to_image(text: str) -> tuple[str, list[str]]:
    """
    🎯 视觉拦截器：捕捉文本中的 Markdown 表格，并实时渲染为图片。
    返回：(清理掉表格的文本，生成的图片绝对路径列表)
    """
    table_pattern = re.compile(r'((?:\|.*\|\n)+\|?(?:[-:]+[-| :]*)\|?\n(?:\|.*\|\n?)+)')
    matches = table_pattern.findall(text)
    
    if not matches:
        return text, []

    image_paths = []
    
    for idx, md_table in enumerate(matches):
        try:
            lines = [line.strip() for line in md_table.strip().split('\n') if '|' in line]
            if len(lines) < 3:
                continue
                
            headers = [x.strip() for x in lines[0].split('|') if x.strip()]
            data = []
            for line in lines[2:]:
                row = [x.strip() for x in line.split('|') if x.strip()]
                if row:
                    data.append(row)
                    
            if not headers or not data:
                continue

            df = pd.DataFrame(data, columns=headers)  # type: ignore[arg-type]
            fig, ax = plt.subplots(figsize=(len(headers) * 1.5 + 1, len(data) * 0.5 + 1))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=df.values, colLabels=df.columns.tolist(), cellLoc='center', loc='center')  # type: ignore[arg-type]
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            img_filename = f"table_render_{int(time.time())}_{idx}.png"
            img_path = (SANDBOX_DIR / img_filename).resolve()
            plt.savefig(img_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
            
            image_paths.append(str(img_path))
            
            text = text.replace(md_table, f"\n\n📊 [表格已渲染为高清图片，请查看上方]\n\n")
            
        except Exception as e:
            print(f"表格渲染拦截失败：{e}")
            continue
            
    return text, image_paths

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
    
    # 交互细节：让机器人在顶部显示“正在输入...”，体验拉满
    effective_chat = update.effective_chat
    if effective_chat is not None:
        await context.bot.send_chat_action(chat_id=effective_chat.id, action='typing')
    
    try:
        # 🚀 唤醒底层 AI 引擎
        response = agent_with_chat_history.invoke(
            {
                "input": user_msg,
                "user_profile": get_user_profile(),
                "current_time": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
            },
            # 物理隔离：用 Telegram 的 User ID 作为独立记忆通道
            config={"configurable": {"session_id": f"tg_session_{user_id}"}}
        )
        
        reply_text = response['output']
        
        # 1. 🎯 触发表格视觉拦截器
        text_without_table, table_images = render_markdown_table_to_image(reply_text)
        
        # 2. 发送渲染好的表格图片
        for img_path in table_images:
            try:
                if update.message is not None:
                    with open(img_path, 'rb') as photo:
                        await update.message.reply_photo(photo=photo)
            except Exception as e:
                logger.error(f"发送表格图片失败：{e}")
                
        # 3. 处理剩下的文本和普通图片 (沿用原有的正则逻辑，但此时的文本是没有 Markdown 表格的)
        # 提取本地图片路径
        img_match = re.search(r'!\[.*?\]\((.*?\.png)\)', text_without_table)
        
        if img_match:
            img_filename = img_match.group(1).replace("./", "")
            img_path = (SANDBOX_DIR / img_filename).resolve()
            
            if img_path.exists() and update.message is not None:
                clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', text_without_table).strip()
                with open(img_path, 'rb') as photo:
                    await update.message.reply_photo(photo=photo, caption=clean_text[:1024])
                return
                
        # 如果没有普通图片，直接发送清洗后的纯文本
        if update.message is not None:
            await update.message.reply_text(text_without_table)
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        message = update.message
        if message is not None:
            await message.reply_text(f"⚠️ 系统熔断：{str(e)}")

def main():
    """启动机器人"""
    logger.info("启动 OmniStock Telegram Bot...")
    
    # 构建应用
    application = Application.builder().token(str(TG_BOT_TOKEN)).build()

    # 注册处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动长轮询，Bot 会一直挂在后台监听
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()