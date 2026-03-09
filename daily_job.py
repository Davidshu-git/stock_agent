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
from valuation_engine import fetch_stock_price_raw


console: Console = Console()


load_dotenv()


def fetch_global_indices() -> str:
    """
    抓取全球核心指数当日涨跌幅数据。

    使用 valuation_engine.fetch_stock_price_raw 获取三大核心指数（沪深 300、恒生指数、纳斯达克 100）的当日行情，
    计算涨跌幅百分比，并提供降级容错机制。

    Returns:
        str: 格式化后的指数涨跌幅文本，格式如：
             "【今日核心指数】沪深 300: +1.25%, 恒生指数：-0.50%, 纳斯达克 100: +0.88%"
    """
    indices_config: Dict[str, Dict[str, str]] = {
        "沪深 300": {"ticker": "000300.SS", "name": "沪深 300"},
        "恒生指数": {"ticker": "^HSI", "name": "恒生指数"},
        "纳斯达克 100": {"ticker": "^NDX", "name": "纳斯达克 100"},
    }

    results: List[str] = []

    for config in indices_config.values():
        ticker: str = config["ticker"]
        name: str = config["name"]

        try:
            price_data = fetch_stock_price_raw(ticker)
            open_price: float = price_data["open"]
            close_price: float = price_data["close"]

            if open_price != 0:
                change_pct: float = ((close_price - open_price) / open_price) * 100
                sign: str = "+" if change_pct >= 0 else ""
                results.append(f"{name}: {sign}{change_pct:.2f}%")
            else:
                results.append(f"{name}: 获取失败")
        except Exception:
            results.append(f"{name}: 获取失败")

    return f"【今日核心指数】{', '.join(results)}"


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


def generate_market_report(news_text: str, user_memory: str, indices_data: str) -> str:
    """
    调用大模型生成盘后市场报告。

    Args:
        news_text: 多源聚合后的全球市场资讯文本。
        user_memory: 用户持仓与偏好记忆字符串。
        indices_data: 全球核心指数当日涨跌幅数据。

    Returns:
        str: 生成的 Markdown 格式报告。
    """
    dashscope_key: str = os.getenv("DASHSCOPE_API_KEY", "")

    if not dashscope_key:
        raise ValueError("DASHSCOPE_API_KEY 未配置")

    system_prompt: str = """
你是一位顶级的全球宏观策略分析师、私人财富管家与极其严苛的风控专家。请根据传入的【200 条全球市场宏观资讯】、【今日核心指数涨跌幅数据】和用户的【个人持仓记忆】，生成一份具备极高专业度的盘后研报。

你的回复必须严格采用 Markdown 格式，并强制包含以下三大核心模块：

### 1. 🌍 全球宏观与三大市场复盘
- 从海量资讯中，分别提炼出 A 股、港股、美股 的绝对核心事件与盘面主线逻辑。
- **请结合提供的【今日核心指数涨跌幅数据】，对三大市场的盘面情绪进行定量与定性结合的精准复盘。**
- 缺失特定市场重大新闻时可一笔带过，切忌废话。

### 2. 🎯 专属持仓洞察
- 将三大市场的热点与用户的跨市场持仓记忆进行深度碰撞，给出具体的调仓、减仓或持有建议。

### 3. ⚠️ 专属风控与黑天鹅预警 (核心重点)
- 强制扮演极度悲观的风控官，从资讯中嗅探可能引发市场剧烈波动的潜在风险。
- **宏观防线**：重点扫描美联储货币政策表态突变、地缘政治摩擦、核心经济数据爆雷等全局性黑天鹅。
- **动态赛道雷达（核心机制）**：必须先深度剖析用户的【个人持仓记忆】，推导出用户当前持仓标的所属的**核心行业与产业链**。然后，在海量资讯中，精准锁定并提取针对这些"特定持仓赛道"的负面异动、政策监管打压或供应链断裂等风险。
- **高亮要求**：一旦发现上述风险，必须使用醒目的加粗和警示符号（如 ❗ 或 🚨）进行高亮。如果当日未监测到明显风险，也必须明确输出"✅ 我们的专属风控雷达今日未发现针对您持仓赛道的重大异动与宏观风险"。
"""

    user_prompt: str = f"""
【今日核心指数涨跌幅数据】
{indices_data}

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

    indices_data: str = fetch_global_indices()
    console.print(f"[bold dim]📊 [指数数据] {indices_data}[/bold dim]")

    try:
        report_content: str = generate_market_report(news_text, user_memory, indices_data)
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
