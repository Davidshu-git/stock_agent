"""
盘后调度主程序 - 每日盘后热点分析与邮件推送调度器。

本模块负责：
1. 定时调度（每日 15:30 盘后）
2. 高可用多源数据聚合（财联社/新浪/东财并行拉取 + 去重融合）
3. 读取用户持仓记忆
4. 调用大模型生成盘后报告
5. 发送邮件推送
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import schedule
import akshare as ak
import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
from rich.console import Console

from notifier import send_market_report_email


console: Console = Console()


load_dotenv()


def load_user_profile() -> str:
    """
    读取用户持仓与偏好记忆文件。

    Returns:
        str: 用户记忆内容，如果文件不存在或解析失败则返回降级提示。
    """
    profile_path: Path = Path("./memory/user_profile.json").resolve()

    if not profile_path.exists():
        return "暂无历史持仓与偏好记录"

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)

        if not data:
            return "暂无历史持仓与偏好记录"

        memory_str: str = "\n".join([f"- 【{k}】: {v}" for k, v in data.items()])
        return memory_str
    except json.JSONDecodeError:
        return "暂无历史持仓与偏好记录"
    except Exception:
        return "暂无历史持仓与偏好记录"


def fetch_global_market_news() -> str:
    """
    高可用多源数据聚合抓取，内置三级降级与数据融合机制。

    并行拉取以下三个宏观资讯源：
    1. 财联社全球电报 (ak.stock_info_global_cls) - 列名：['标题', '内容', '发布日期', '发布时间']
    2. 新浪 7x24 快讯 (ak.stock_info_global_sina) - 列名：['时间', '内容']
    3. 东财全球快讯 (ak.stock_info_global_em) - 列名：['标题', '摘要', '发布时间', '链接']

    Returns:
        str: 合并去重后的资讯文本（最多 200 条）。

    Raises:
        RuntimeError: 所有数据源全部失效时抛出。
    """
    data_sources: List[Dict[str, Any]] = [
        {"name": "财联社全球电报", "func": lambda: ak.stock_info_global_cls(symbol="全部")},
        {"name": "新浪 7x24", "func": lambda: ak.stock_info_global_sina()},
        {"name": "东财全球快讯", "func": lambda: ak.stock_info_global_em()},
    ]

    column_mapping: Dict[str, Dict[str, str]] = {
        "财联社全球电报": {"time": "发布时间", "content": "内容"},
        "新浪 7x24": {"time": "时间", "content": "内容"},
        "东财全球快讯": {"time": "发布时间", "content": "摘要"},
    }

    dfs: List[pd.DataFrame] = []

    for source in data_sources:
        source_name: str = source["name"]
        fetch_func = source["func"]

        try:
            df: pd.DataFrame = fetch_func()

            if df is not None and not df.empty:
                console.print(f"[bold green]✅ 成功获取 {source_name}: {len(df)} 条数据[/bold green]")

                mapping = column_mapping.get(source_name, {})
                time_col: str = mapping.get("time", "")
                content_col: str = mapping.get("content", "")

                if time_col in df.columns and content_col in df.columns:
                    selected_df: pd.DataFrame = df[[time_col, content_col]].copy()
                    selected_df.columns = ["time", "content"]
                    dfs.append(selected_df)
                    console.print(f"[dim]   └─ 列名映射：{time_col} → time, {content_col} → content[/dim]")
                else:
                    console.print(f"[bold red]❌ {source_name} 列名不匹配，期望 time='{time_col}', content='{content_col}'[/bold red]")
                    console.print(f"[dim]   实际列名：{list(df.columns)}[/dim]")
                    continue
            else:
                console.print(f"[bold yellow]⚠️  {source_name} 返回空数据[/bold yellow]")
        except Exception as e:
            console.print(f"[bold red]❌ {source_name} 获取失败：{type(e).__name__} - {str(e)}[/bold red]")
            continue

    if not dfs:
        raise RuntimeError("所有宏观资讯源全部失效")

    merged_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    deduped_df: pd.DataFrame = merged_df.drop_duplicates(subset=["content"]).copy()

    if "time" in deduped_df.columns:
        try:
            deduped_df.loc[:, "time"] = pd.to_datetime(deduped_df["time"], errors="coerce")
            deduped_df = deduped_df.sort_values("time", ascending=False).reset_index(drop=True)
            console.print("[dim]   └─ 时间排序成功（已转换为 datetime）[/dim]")
        except Exception as e:
            console.print(f"[yellow dim]⚠️  时间排序失败：{type(e).__name__}，使用原始顺序[/yellow dim]")

    final_df: pd.DataFrame = deduped_df.head(200).copy()

    console.print(f"[bold cyan]📊 多源聚合去重完成，最终采用 {len(final_df)} 条有效资讯进行推理。[/bold cyan]")

    news_items: List[str] = []
    for _, row in final_df.iterrows():
        time_str: str = str(row.get("time", ""))
        content_str: str = str(row.get("content", ""))
        if content_str.strip():
            news_items.append(f"[{time_str}] {content_str}")

    return "\n".join(news_items)


def generate_market_report(news_text: str, user_memory: str) -> str:
    """
    调用大模型生成盘后市场报告。

    Args:
        news_text: 多源聚合后的全球市场资讯文本。
        user_memory: 用户持仓与偏好记忆字符串。

    Returns:
        str: 生成的 Markdown 格式报告。
    """
    dashscope_key: str = os.getenv("DASHSCOPE_API_KEY", "")

    if not dashscope_key:
        raise ValueError("DASHSCOPE_API_KEY 未配置")

    system_prompt: str = """
你是一位顶级的全球宏观策略分析师与私人财富管家。请根据【全球市场资讯】和用户的【个人持仓记忆】，生成盘后报告。

任务要求：
1. 从海量资讯中，分别提炼出 A 股、港股、美股 三大市场的绝对核心事件与盘面逻辑。
   - 如果某个市场缺失重大新闻，可以一笔带过。
   - 每个市场提取 1-3 个核心主线，包含"核心逻辑"与"盘面表现"。
2. 单独开辟一个《专属持仓洞察》模块，将三大市场的热点与用户的跨市场持仓进行碰撞，给出具体的仓位管理建议。
3. 输出格式严格遵守 Markdown，使用标题、列表、加粗等排版。
"""

    user_prompt: str = f"""
【全球市场资讯】
{news_text}

【个人持仓与偏好记忆】
{user_memory}

请生成今日全球盘后报告：
"""

    llm: ChatOpenAI = ChatOpenAI(
        model="qwen3.5-plus",
        base_url="https://coding.dashscope.aliyuncs.com/v1",
        temperature=0,
        timeout=120,  # 生产环境：长文本推理需要充足时间 (200 条资讯 + 跨市场分析)
        max_retries=3,
        api_key=SecretStr(dashscope_key),
    )

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )

    content: str = response.content if isinstance(response.content, str) else str(response.content)
    return content


def job_routine() -> None:
    """
    盘后调度主流程：获取数据 -> 生成报告 -> 发送邮件。
    """
    console.print(f"\n[bold cyan]⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始执行盘后调度任务...[/bold cyan]")

    try:
        news_text: str = fetch_global_market_news()
    except RuntimeError as e:
        console.print(f"[bold red]❌ [调度任务] {str(e)}[/bold red]")
        return

    user_memory: str = load_user_profile()
    console.print(f"[bold dim]🧠 [记忆读取] 用户记忆加载完成[/bold dim]")

    try:
        report_content: str = generate_market_report(news_text, user_memory)
    except ValueError as e:
        console.print(f"[bold red]❌ [报告生成] {str(e)}[/bold red]")
        return
    except Exception:
        console.print("[bold red]❌ [报告生成] 大模型推理失败，任务中止。[/bold red]")
        return

    console.print("[bold green]✔️  [报告生成] 盘后报告生成完毕[/bold green]")

    subject: str = f"盘后报告 | {datetime.now().strftime('%Y-%m-%d')}"

    try:
        send_market_report_email(subject, report_content)
        console.print("[bold green]📧 [邮件推送] 报告已成功发送[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ [邮件推送] 发送失败：{type(e).__name__} - {str(e)}[/bold red]")

    console.print(f"[bold cyan]✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 盘后调度任务执行完毕[/bold cyan]\n")


def run_scheduler() -> None:
    """
    启动定时调度器，进入挂起等待状态。
    """
    schedule.every().day.at("15:30").do(job_routine)

    console.print("[bold cyan]🕒 调度器已启动，等待每日 15:30 执行盘后任务...[/bold cyan]")
    console.print("[dim]按 Ctrl+C 停止调度器[/dim]")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️  调度器已停止[/bold yellow]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--test", "-t"]:
        console.print("[bold magenta]🧪 测试模式：立即执行一次盘后调度任务...[/bold magenta]\n")
        job_routine()
    else:
        run_scheduler()
