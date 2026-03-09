"""
估值引擎核心模块 - 纯 Python 量化查价与绘图底层逻辑。

本模块提供：
1. 全球股票价格查询（支持美股/A 股/港股）
2. A 股 ETF 价格查询（双源降级）
3. K 线图生成
4. 持仓估值计算
"""

import yfinance as yf
import akshare as ak
import mplfinance as mpf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


def format_universal_ticker(ticker: str) -> str:
    """
    智能推断股票市场并格式化为 yfinance 识别的代码。
    
    Args:
        ticker: 原始股票代码（如 AAPL, 600519, 0700）
    
    Returns:
        str: 格式化后的 ticker（如 AAPL, 600519.SS, 0700.HK）
    """
    ticker = ticker.strip().upper()
    
    if "." in ticker:
        return ticker
        
    if ticker.isalpha():
        return ticker
        
    digits = ''.join(filter(str.isdigit, ticker))
    
    if len(digits) <= 4 and digits:
        return f"{digits.zfill(4)}.HK"
        
    if len(digits) == 6:
        if digits.startswith(('60', '68')):
            return f"{digits}.SS"
        else:
            return f"{digits}.SZ"
            
    return ticker


def fetch_stock_price_raw(ticker: str, date: Optional[str] = None) -> Dict[str, Any]:
    """
    获取全球股票原始价格数据（支持美股/A 股/港股）。
    
    Args:
        ticker: 股票代码（如 AAPL, 600519, 0700）
        date: 可选日期 'YYYY-MM-DD'，未提供则返回最近交易日
    
    Returns:
        dict: {"open": xxx, "close": xxx, "date": "...", "high": xxx, "low": xxx}
    
    Raises:
        ValueError: 日期格式不正确
        KeyError: 数据字段缺失
        IndexError: 无历史数据
    """
    formatted_ticker = format_universal_ticker(ticker)
    stock = yf.Ticker(formatted_ticker)
    
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            next_date = target_date + timedelta(days=1)
            hist = stock.history(
                start=target_date.strftime("%Y-%m-%d"),
                end=next_date.strftime("%Y-%m-%d")
            )
            date_label = date
        except ValueError as e:
            raise ValueError(f"日期格式不正确：{e}")
    else:
        hist = stock.history(period="1d")
        date_label = datetime.now().strftime("%Y-%m-%d")
    
    if hist.empty:
        raise IndexError(f"未找到 {formatted_ticker} 的历史数据")
    
    open_val = hist['Open'].iloc[0]
    close_val = hist['Close'].iloc[0]
    high_val = hist['High'].iloc[0] if 'High' in hist.columns else None
    low_val = hist['Low'].iloc[0] if 'Low' in hist.columns else None
    
    if pd.isna(open_val) or pd.isna(close_val):
        raise ValueError("数据不完整（存在空值）")
    
    actual_date = hist.index[0].strftime("%Y-%m-%d")
    
    result: Dict[str, Any] = {
        "ticker": formatted_ticker,
        "open": round(float(open_val), 2),
        "close": round(float(close_val), 2),
        "date": actual_date,
        "query_date": date_label
    }
    
    if high_val and not pd.isna(high_val):
        result["high"] = round(float(high_val), 2)
    if low_val and not pd.isna(low_val):
        result["low"] = round(float(low_val), 2)
    
    return result


def fetch_etf_price_raw(etf_code: str, date: Optional[str] = None) -> Dict[str, Any]:
    """
    获取 A 股 ETF 原始价格数据（yfinance + akshare 双源降级）。
    
    Args:
        etf_code: 6 位 ETF 代码（如 '513050'）
        date: 可选日期 'YYYY-MM-DD'，未提供则返回实时行情
    
    Returns:
        dict: {
            "etf_code": "513050",
            "open": xxx, "close": xxx, "high": xxx, "low": xxx,
            "volume": xxx, "date": "...",
            "source": "yfinance" | "akshare"
        }
    
    Raises:
        ValueError: ETF 代码格式不正确或数据不完整
        RuntimeError: 所有数据源均失败
    """
    etf_code = etf_code.strip()
    if not etf_code.isdigit() or len(etf_code) != 6:
        raise ValueError("ETF 代码格式不正确，请输入 6 位数字代码")
    
    if etf_code.startswith(('50', '51', '58')):
        suffix = '.SS'
    elif etf_code.startswith(('15', '16')):
        suffix = '.SZ'
    else:
        suffix = ''
    
    formatted_code = etf_code + suffix if suffix else etf_code
    
    yf_error: Optional[Exception] = None
    
    try:
        stock = yf.Ticker(formatted_code)
        
        if date:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            next_date = target_date + timedelta(days=1)
            hist = stock.history(
                start=target_date.strftime("%Y-%m-%d"),
                end=next_date.strftime("%Y-%m-%d")
            )
            date_label = date
        else:
            hist = stock.history(period="1d")
            date_label = datetime.now().strftime("%Y-%m-%d")
        
        if not hist.empty:
            open_val = hist['Open'].iloc[0]
            close_val = hist['Close'].iloc[0]
            
            if pd.isna(open_val) or pd.isna(close_val):
                raise ValueError("yfinance 数据不完整")
            
            return {
                "etf_code": etf_code,
                "ticker": formatted_code,
                "open": round(float(open_val), 3),
                "close": round(float(close_val), 3),
                "high": round(float(hist['High'].iloc[0]), 3),
                "low": round(float(hist['Low'].iloc[0]), 3),
                "volume": int(hist['Volume'].iloc[0]),
                "date": hist.index[0].strftime("%Y-%m-%d"),
                "query_date": date_label,
                "source": "yfinance"
            }
    except Exception as e:
        yf_error = e
    
    try:
        if date:
            df = ak.fund_etf_hist_em(
                symbol=etf_code,
                period="daily",
                start_date=date.replace('-', ''),
                end_date=date.replace('-', ''),
                adjust=""
            )
            if df is None or df.empty:
                raise ValueError(f"akshare 未找到 {etf_code} 在 {date} 的数据")
            
            open_val = df['开盘'].iloc[0]
            close_val = df['收盘'].iloc[0]
            
            if pd.isna(open_val) or pd.isna(close_val):
                raise ValueError("akshare 数据不完整")
            
            return {
                "etf_code": etf_code,
                "open": round(float(open_val), 3),
                "close": round(float(close_val), 3),
                "high": round(float(df['最高'].iloc[0]), 3),
                "low": round(float(df['最低'].iloc[0]), 3),
                "volume": int(df['成交量'].iloc[0]),
                "date": date,
                "query_date": date,
                "source": "akshare_hist"
            }
        else:
            all_etfs = ak.fund_etf_spot_em()
            df = all_etfs[all_etfs['代码'] == etf_code]
            
            if df is None or df.empty:
                raise ValueError(f"akshare 未找到 ETF {etf_code} 的实时行情")
            
            return {
                "etf_code": etf_code,
                "current_price": round(float(df['最新价'].iloc[0]), 3),
                "open": round(float(df['今开'].iloc[0]), 3),
                "high": round(float(df['最高'].iloc[0]), 3),
                "low": round(float(df['最低'].iloc[0]), 3),
                "prev_close": round(float(df['昨收'].iloc[0]), 3),
                "change_percent": round(float(df['涨跌幅'].iloc[0]), 2),
                "volume": int(df['成交量'].iloc[0]),
                "amount": round(float(df['成交额'].iloc[0]) / 10000, 2),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "query_date": "实时",
                "source": "akshare_spot"
            }
    except Exception as ak_error:
        if yf_error:
            raise RuntimeError(f"所有数据源失败：yfinance={yf_error}, akshare={ak_error}")
        raise RuntimeError(f"akshare 查询失败：{ak_error}")


def generate_kline_chart(ticker: str, save_dir: Path) -> Dict[str, Any]:
    """
    生成股票 30 天 K 线走势图。
    
    Args:
        ticker: 股票代码
        save_dir: 图片保存目录
    
    Returns:
        dict: {
            "max_price": xxx,
            "min_price": xxx,
            "latest_close": xxx,
            "file_name": "...",
            "file_path": "..."
        }
    
    Raises:
        IndexError: 无历史数据
        KeyError: 数据字段缺失
    """
    formatted_ticker = format_universal_ticker(ticker)
    stock = yf.Ticker(formatted_ticker)
    hist = stock.history(period="1mo")
    
    if hist.empty:
        raise IndexError(f"未找到 {formatted_ticker} 的历史数据，无法绘图")
    
    safe_name = formatted_ticker.replace('.', '_')
    chart_filename = f"{safe_name}_30d_chart.png"
    chart_path = (save_dir / chart_filename).resolve()
    
    mpf.plot(
        hist, type='candle', volume=True, style='yahoo',
        title=f"{formatted_ticker} 30-Day Trend", mav=(5, 10),
        savefig=str(chart_path)
    )
    
    max_price = round(float(hist['High'].max()), 2)
    min_price = round(float(hist['Low'].min()), 2)
    latest_close = round(float(hist['Close'].iloc[-1]), 2)
    
    return {
        "ticker": formatted_ticker,
        "max_price": max_price,
        "min_price": min_price,
        "latest_close": latest_close,
        "file_name": chart_filename,
        "file_path": str(chart_path)
    }


def calculate_portfolio_valuation(positions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算持仓组合的精确总市值与今日总盈亏。
    
    Args:
        positions: 持仓字典，格式如：
            {
                "AAPL": {"shares": 100, "cost_basis": 150.0},
                "600519": {"shares": 50, "cost_basis": 1800.0},
                "513050": {"shares": 1000, "cost_basis": 1.2, "type": "etf"}
            }
    
    Returns:
        dict: {
            "total_market_value": xxx,
            "total_cost": xxx,
            "total_profit_loss": xxx,
            "profit_loss_percent": xxx,
            "holdings": [
                {
                    "ticker": "AAPL",
                    "shares": 100,
                    "current_price": 175.0,
                    "market_value": 17500.0,
                    "profit_loss": 2500.0,
                    "profit_loss_percent": 16.67
                },
                ...
            ]
        }
    """
    holdings_result = []
    total_market_value = 0.0
    total_cost = 0.0
    
    for ticker, position in positions.items():
        shares = position.get("shares", 0)
        cost_basis = position.get("cost_basis", 0)
        is_etf = position.get("type", "stock") == "etf"
        
        try:
            if is_etf:
                price_data = fetch_etf_price_raw(ticker)
                current_price = price_data.get("current_price", price_data.get("close"))
            else:
                price_data = fetch_stock_price_raw(ticker)
                current_price = price_data["close"]
            
            market_value = current_price * shares
            cost_value = cost_basis * shares
            profit_loss = market_value - cost_value
            profit_loss_percent = (profit_loss / cost_value * 100) if cost_value != 0 else 0
            
            holdings_result.append({
                "ticker": ticker,
                "shares": shares,
                "current_price": current_price,
                "market_value": round(market_value, 2),
                "cost_basis": cost_basis,
                "cost_value": round(cost_value, 2),
                "profit_loss": round(profit_loss, 2),
                "profit_loss_percent": round(profit_loss_percent, 2)
            })
            
            total_market_value += market_value
            total_cost += cost_value
            
        except Exception as e:
            holdings_result.append({
                "ticker": ticker,
                "shares": shares,
                "error": f"获取价格失败：{type(e).__name__} - {str(e)}"
            })
    
    total_profit_loss = total_market_value - total_cost
    total_profit_loss_percent = (total_profit_loss / total_cost * 100) if total_cost != 0 else 0
    
    return {
        "total_market_value": round(total_market_value, 2),
        "total_cost": round(total_cost, 2),
        "total_profit_loss": round(total_profit_loss, 2),
        "profit_loss_percent": round(total_profit_loss_percent, 2),
        "holdings": holdings_result,
        "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
