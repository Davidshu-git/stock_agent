"""
估值引擎单元测试模块。

本模块提供对 calculate_portfolio_valuation 函数的完整测试覆盖，
包括正常场景、边界情况和异常处理。
"""

from typing import Any, Dict
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from valuation_engine import calculate_portfolio_valuation, fetch_exchange_rates, fetch_stock_price_raw, fetch_etf_price_raw


class TestCalculatePortfolioValuationHappyPath:
    """测试正常计算逻辑 - 多币种持仓场景。"""

    @patch('valuation_engine.fetch_etf_price_raw')
    @patch('valuation_engine.fetch_stock_price_raw')
    @patch('valuation_engine.fetch_exchange_rates')
    def test_multi_currency_portfolio_calculation(
        self,
        mock_fetch_rates: MagicMock,
        mock_fetch_stock: MagicMock,
        mock_fetch_etf: MagicMock
    ) -> None:
        """
        测试多币种持仓组合的精确计算。
        
        Mock 设置:
        - AAPL (美股): 现价 200 USD, 100 股，成本 150 USD
        - 0700.HK (港股): 现价 300 HKD, 50 股，成本 280 HKD
        - 513050 (A 股 ETF): 现价 1.5 CNY, 1000 股，成本 1.2 CNY
        
        断言:
        - 各标的原生市值和折合人民币市值计算精确
        - 总市值和总盈亏等于各标的折算后之和
        """
        # 设置 Mock 返回值
        mock_fetch_rates.return_value = {
            "USD_CNY": 7.0,
            "HKD_CNY": 0.9,
            "CNY_CNY": 1.0
        }

        # 配置股票价格 Mock
        def stock_price_side_effect(ticker: str, date: str | None = None) -> Dict[str, Any]:
            ticker_upper = ticker.upper()
            if "AAPL" in ticker_upper:
                return {
                    "ticker": "AAPL",
                    "open": 198.0,
                    "close": 200.0,
                    "high": 202.0,
                    "low": 197.0,
                    "date": "2026-03-09"
                }
            elif "0700.HK" in ticker_upper:
                return {
                    "ticker": "0700.HK",
                    "open": 298.0,
                    "close": 300.0,
                    "high": 305.0,
                    "low": 295.0,
                    "date": "2026-03-09"
                }
            else:
                raise ValueError(f"Unknown ticker: {ticker}")

        mock_fetch_stock.side_effect = stock_price_side_effect

        # 配置 ETF 价格 Mock
        mock_fetch_etf.return_value = {
            "etf_code": "513050",
            "ticker": "513050.SS",
            "open": 1.48,
            "close": 1.5,
            "high": 1.52,
            "low": 1.47,
            "volume": 1000000,
            "date": "2026-03-09",
            "current_price": 1.5,
            "source": "yfinance"
        }

        # 准备输入持仓数据
        positions: Dict[str, Dict[str, Any]] = {
            "AAPL": {"shares": 100, "cost_basis": 150.0},
            "0700.HK": {"shares": 50, "cost_basis": 280.0},
            "513050": {"shares": 1000, "cost_basis": 1.2, "type": "etf"}
        }

        # 执行计算
        result = calculate_portfolio_valuation(positions)

        # 验证返回值结构
        assert "total_market_value" in result
        assert "total_cost" in result
        assert "total_profit_loss" in result
        assert "holdings" in result
        assert "exchange_rates" in result

        # 验证汇率 Mock 被调用
        mock_fetch_rates.assert_called_once()

        # 查找各标的计算结果
        holdings_by_ticker = {h["ticker"]: h for h in result["holdings"]}

        # ========== 断言 AAPL 计算结果 ==========
        aapl = holdings_by_ticker["AAPL"]
        assert aapl["shares"] == 100
        assert aapl["current_price"] == 200.0
        assert aapl["currency"] == "USD"
        assert aapl["exchange_rate"] == 7.0

        # 原生市值 = 200 * 100 = 20000 USD
        expected_aapl_native_mv = 200.0 * 100
        assert aapl["native_market_value"] == pytest.approx(expected_aapl_native_mv, rel=1e-2)

        # 折合人民币市值 = 20000 * 7.0 = 140000 CNY
        expected_aapl_cny_mv = expected_aapl_native_mv * 7.0
        assert aapl["market_value_cny"] == pytest.approx(expected_aapl_cny_mv, rel=1e-2)

        # ========== 断言 0700.HK 计算结果 ==========
        hk_stock = holdings_by_ticker["0700.HK"]
        assert hk_stock["shares"] == 50
        assert hk_stock["current_price"] == 300.0
        assert hk_stock["currency"] == "HKD"
        assert hk_stock["exchange_rate"] == 0.9

        # 原生市值 = 300 * 50 = 15000 HKD
        expected_hk_native_mv = 300.0 * 50
        assert hk_stock["native_market_value"] == pytest.approx(expected_hk_native_mv, rel=1e-2)

        # 折合人民币市值 = 15000 * 0.9 = 13500 CNY
        expected_hk_cny_mv = expected_hk_native_mv * 0.9
        assert hk_stock["market_value_cny"] == pytest.approx(expected_hk_cny_mv, rel=1e-2)

        # ========== 断言 513050 ETF 计算结果 ==========
        etf = holdings_by_ticker["513050"]
        assert etf["shares"] == 1000
        assert etf["current_price"] == 1.5
        assert etf["currency"] == "CNY"
        assert etf["exchange_rate"] == 1.0

        # 原生市值 = 1.5 * 1000 = 1500 CNY
        expected_etf_native_mv = 1.5 * 1000
        assert etf["native_market_value"] == pytest.approx(expected_etf_native_mv, rel=1e-2)

        # 折合人民币市值 = 1500 * 1.0 = 1500 CNY
        expected_etf_cny_mv = expected_etf_native_mv * 1.0
        assert etf["market_value_cny"] == pytest.approx(expected_etf_cny_mv, rel=1e-2)

        # ========== 断言总计 ==========
        # 总市值 = 140000 + 13500 + 1500 = 155000 CNY
        expected_total_mv = expected_aapl_cny_mv + expected_hk_cny_mv + expected_etf_cny_mv
        assert result["total_market_value"] == pytest.approx(expected_total_mv, rel=1e-2)

        # 总成本计算
        expected_total_cost = (
            150.0 * 100 * 7.0 +      # AAPL: 105000 CNY
            280.0 * 50 * 0.9 +       # 0700.HK: 12600 CNY
            1.2 * 1000 * 1.0         # 513050: 1200 CNY
        )
        assert result["total_cost"] == pytest.approx(expected_total_cost, rel=1e-2)

        # 总盈亏 = 总市值 - 总成本
        expected_total_pnl = expected_total_mv - expected_total_cost
        assert result["total_profit_loss"] == pytest.approx(expected_total_pnl, rel=1e-2)


class TestEdgeCasesAndErrorHandling:
    """测试边界情况与异常处理。"""

    @patch('valuation_engine.fetch_stock_price_raw')
    @patch('valuation_engine.fetch_exchange_rates')
    def test_empty_positions(self, mock_fetch_rates: MagicMock, mock_fetch_stock: MagicMock) -> None:
        """
        测试空持仓输入。
        
        断言:
        - 返回的总市值和总盈亏均为 0.0
        - 程序不崩溃，返回合法结构
        """
        mock_fetch_rates.return_value = {
            "USD_CNY": 7.0,
            "HKD_CNY": 0.9,
            "CNY_CNY": 1.0
        }

        positions: Dict[str, Dict[str, Any]] = {}

        result = calculate_portfolio_valuation(positions)

        # 断言总计为零
        assert result["total_market_value"] == 0.0
        assert result["total_cost"] == 0.0
        assert result["total_profit_loss"] == 0.0
        assert result["profit_loss_percent"] == 0.0

        # 断言持仓列表为空
        assert result["holdings"] == []

        # 断言汇率仍被获取
        mock_fetch_rates.assert_called_once()

    @patch('valuation_engine.fetch_etf_price_raw')
    @patch('valuation_engine.fetch_stock_price_raw')
    @patch('valuation_engine.fetch_exchange_rates')
    def test_partial_failure_graceful_degradation(
        self,
        mock_fetch_rates: MagicMock,
        mock_fetch_stock: MagicMock,
        mock_fetch_etf: MagicMock
    ) -> None:
        """
        测试部分标的查价失败时的容错降级。
        
        Mock 设置:
        - AAPL 正常返回价格
        - 0700.HK 抛出异常
        
        断言:
        - AAPL 正常计算
        - 0700.HK 带有 error 字段
        - 总计只包含 AAPL 的贡献
        """
        mock_fetch_rates.return_value = {
            "USD_CNY": 7.0,
            "HKD_CNY": 0.9,
            "CNY_CNY": 1.0
        }

        def stock_price_side_effect(ticker: str, date: str | None = None) -> Dict[str, Any]:
            ticker_upper = ticker.upper()
            if "AAPL" in ticker_upper:
                return {
                    "ticker": "AAPL",
                    "open": 198.0,
                    "close": 200.0,
                    "high": 202.0,
                    "low": 197.0,
                    "date": "2026-03-09"
                }
            elif "0700.HK" in ticker_upper:
                raise Exception("Network timeout: Unable to fetch price for 0700.HK")
            else:
                raise ValueError(f"Unknown ticker: {ticker}")

        mock_fetch_stock.side_effect = stock_price_side_effect

        positions: Dict[str, Dict[str, Any]] = {
            "AAPL": {"shares": 100, "cost_basis": 150.0},
            "0700.HK": {"shares": 50, "cost_basis": 280.0}
        }

        result = calculate_portfolio_valuation(positions)

        # 断言返回结构完整
        assert "holdings" in result
        assert len(result["holdings"]) == 2

        holdings_by_ticker = {h["ticker"]: h for h in result["holdings"]}

        # 断言 AAPL 正常计算
        aapl = holdings_by_ticker["AAPL"]
        assert "error" not in aapl
        assert aapl["current_price"] == 200.0
        assert aapl["market_value_cny"] == 140000.0  # 200 * 100 * 7.0

        # 断言 0700.HK 带有 error 字段
        hk_stock = holdings_by_ticker["0700.HK"]
        assert "error" in hk_stock
        assert "Network timeout" in hk_stock["error"]
        assert hk_stock["shares"] == 50

        # 断言总计只包含 AAPL 的贡献（0700.HK 失败不计入）
        assert result["total_market_value"] == 140000.0
        assert result["total_cost"] == 105000.0  # 150 * 100 * 7.0
        assert result["total_profit_loss"] == 35000.0  # 140000 - 105000

        # 验证 ETF Mock 未被调用（因为没有 ETF 持仓）
        mock_fetch_etf.assert_not_called()

    @patch('valuation_engine.fetch_stock_price_raw')
    @patch('valuation_engine.fetch_exchange_rates')
    def test_zero_shares_position(self, mock_fetch_rates: MagicMock, mock_fetch_stock: MagicMock) -> None:
        """
        测试零股数持仓边界情况。
        
        断言:
        - 市值为 0
        - 程序不崩溃
        """
        mock_fetch_rates.return_value = {
            "USD_CNY": 7.0,
            "HKD_CNY": 0.9,
            "CNY_CNY": 1.0
        }

        mock_fetch_stock.return_value = {
            "ticker": "AAPL",
            "open": 198.0,
            "close": 200.0,
            "high": 202.0,
            "low": 197.0,
            "date": "2026-03-09"
        }

        positions: Dict[str, Dict[str, Any]] = {
            "AAPL": {"shares": 0, "cost_basis": 150.0}
        }

        result = calculate_portfolio_valuation(positions)

        assert result["total_market_value"] == 0.0
        assert result["total_cost"] == 0.0
        assert result["total_profit_loss"] == 0.0

        holding = result["holdings"][0]
        assert holding["native_market_value"] == 0.0
        assert holding["market_value_cny"] == 0.0

    @patch('valuation_engine.fetch_stock_price_raw')
    @patch('valuation_engine.fetch_exchange_rates')
    def test_zero_cost_basis_position(self, mock_fetch_rates: MagicMock, mock_fetch_stock: MagicMock) -> None:
        """
        测试零成本价持仓（赠股或特殊场景）。
        
        断言:
        - 盈亏率为 0（避免除以零）
        - 程序不崩溃
        """
        mock_fetch_rates.return_value = {
            "USD_CNY": 7.0,
            "HKD_CNY": 0.9,
            "CNY_CNY": 1.0
        }

        mock_fetch_stock.return_value = {
            "ticker": "AAPL",
            "open": 198.0,
            "close": 200.0,
            "high": 202.0,
            "low": 197.0,
            "date": "2026-03-09"
        }

        positions: Dict[str, Dict[str, Any]] = {
            "AAPL": {"shares": 100, "cost_basis": 0.0}
        }

        result = calculate_portfolio_valuation(positions)

        holding = result["holdings"][0]
        # 零成本时盈亏率应为 0（代码中已处理）
        assert holding["profit_loss_percent"] == 0.0
        assert holding["profit_loss_cny"] > 0  # 但绝对盈亏应有值

        # 总计的盈亏率也应为 0
        assert result["profit_loss_percent"] == 0.0


class TestExchangeRatesMock:
    """测试汇率 Mock 的正确性。"""

    @patch('valuation_engine.fetch_exchange_rates')
    def test_custom_exchange_rates(self, mock_fetch_rates: MagicMock) -> None:
        """
        测试自定义汇率设置。
        
        断言:
        - Mock 汇率被正确使用
        """
        custom_rates = {
            "USD_CNY": 7.25,
            "HKD_CNY": 0.93,
            "CNY_CNY": 1.0
        }
        mock_fetch_rates.return_value = custom_rates

        with patch('valuation_engine.fetch_stock_price_raw') as mock_fetch_stock:
            mock_fetch_stock.return_value = {
                "ticker": "AAPL",
                "open": 198.0,
                "close": 200.0,
                "high": 202.0,
                "low": 197.0,
                "date": "2026-03-09"
            }

            positions = {"AAPL": {"shares": 100, "cost_basis": 150.0}}
            result = calculate_portfolio_valuation(positions)

            # 使用自定义汇率 7.25 计算
            expected_mv = 200.0 * 100 * 7.25
            assert result["total_market_value"] == pytest.approx(expected_mv, rel=1e-2)
            assert result["exchange_rates"] == custom_rates
