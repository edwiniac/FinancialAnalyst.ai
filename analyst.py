#!/usr/bin/env python3
"""
FinancialAnalyst.ai - Autonomous AI agent for financial analysis.

An intelligent agent that combines LLM reasoning with financial data
to provide investment research and portfolio analysis.
"""

import json
import os
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field

import yfinance as yf
from openai import OpenAI
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


@dataclass
class AnalysisResult:
    """Structured analysis result."""
    symbol: str
    summary: str
    recommendation: str
    confidence: str
    key_metrics: dict
    technical_view: dict
    risks: list[str]
    catalysts: list[str]
    raw_data: dict = field(default_factory=dict)


class FinancialTools:
    """Financial data tools for the agent."""

    @staticmethod
    def get_stock_price(symbol: str, include_history: bool = False) -> dict:
        """Get current stock price and basic info."""
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        result = {
            "symbol": symbol.upper(),
            "name": info.get("shortName", info.get("longName", symbol)),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "open": info.get("open") or info.get("regularMarketOpen"),
            "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

        if result["price"] and result["previous_close"]:
            change = result["price"] - result["previous_close"]
            change_percent = (change / result["previous_close"]) * 100
            result["change"] = round(change, 2)
            result["change_percent"] = round(change_percent, 2)

        if include_history:
            hist = ticker.history(period="5d")
            if not hist.empty:
                result["history"] = [
                    {"date": idx.strftime("%Y-%m-%d"), "close": round(row["Close"], 2)}
                    for idx, row in hist.iterrows()
                ]

        return result

    @staticmethod
    def get_company_info(symbol: str) -> dict:
        """Get detailed company information."""
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        return {
            "symbol": symbol.upper(),
            "name": info.get("longName"),
            "description": info.get("longBusinessSummary"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country"),
            "website": info.get("website"),
            "financials": {
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
            },
            "ratios": {
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "peg": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "debt_to_equity": info.get("debtToEquity"),
            },
            "growth": {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
            },
            "dividends": {
                "yield": info.get("dividendYield"),
                "rate": info.get("dividendRate"),
            },
            "analyst": {
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "recommendation": info.get("recommendationKey"),
            }
        }

    @staticmethod
    def get_technical_analysis(symbol: str) -> dict:
        """Get technical analysis indicators."""
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period="6mo")

        if hist.empty:
            return {"error": f"No data available for {symbol}"}

        close = hist["Close"]
        current_price = close.iloc[-1]

        # Moving averages
        ma_20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else None
        ma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
        ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else None

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # Support/Resistance
        recent_high = hist["High"].tail(20).max()
        recent_low = hist["Low"].tail(20).min()

        # Trend
        if ma_20 and ma_50:
            if current_price > ma_20 > ma_50:
                trend = "Bullish"
            elif current_price < ma_20 < ma_50:
                trend = "Bearish"
            else:
                trend = "Neutral"
        else:
            trend = "Insufficient data"

        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100

        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "trend": trend,
            "moving_averages": {
                "ma_20": round(ma_20, 2) if ma_20 else None,
                "ma_50": round(ma_50, 2) if ma_50 else None,
                "ma_200": round(ma_200, 2) if ma_200 else None,
                "above_ma20": current_price > ma_20 if ma_20 else None,
                "above_ma50": current_price > ma_50 if ma_50 else None,
            },
            "momentum": {
                "rsi_14": round(current_rsi, 2) if current_rsi else None,
                "rsi_signal": "Overbought" if current_rsi and current_rsi > 70 else "Oversold" if current_rsi and current_rsi < 30 else "Neutral",
                "macd": round(macd_line.iloc[-1], 4) if not macd_line.empty else None,
                "macd_signal": round(signal_line.iloc[-1], 4) if not signal_line.empty else None,
            },
            "levels": {
                "resistance": round(recent_high, 2),
                "support": round(recent_low, 2),
            },
            "volatility_annual": f"{volatility:.1f}%",
        }

    @staticmethod
    def get_news(symbol: str, limit: int = 5) -> dict:
        """Get recent news for a stock."""
        ticker = yf.Ticker(symbol.upper())
        news = ticker.news[:limit] if ticker.news else []

        articles = []
        for item in news:
            articles.append({
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "published": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else None,
            })

        return {"symbol": symbol.upper(), "articles": articles}

    @staticmethod
    def compare_stocks(symbols: list[str]) -> dict:
        """Compare multiple stocks."""
        comparisons = []
        for symbol in symbols[:5]:  # Limit to 5
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            comparisons.append({
                "symbol": symbol.upper(),
                "name": info.get("shortName"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg": info.get("pegRatio"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "dividend_yield": info.get("dividendYield"),
            })
        return {"comparison": comparisons}


class FinancialAnalyst:
    """The main AI agent for financial analysis."""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get current stock price, daily change, and basic stats",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "include_history": {"type": "boolean", "default": False}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_company_info",
                "description": "Get detailed company info including financials, ratios, and analyst targets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_technical_analysis",
                "description": "Get technical indicators: moving averages, RSI, MACD, support/resistance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": "Get recent news articles for a stock",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "limit": {"type": "integer", "default": 5}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_stocks",
                "description": "Compare multiple stocks side by side",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ticker symbols to compare"
                        }
                    },
                    "required": ["symbols"]
                }
            }
        }
    ]

    SYSTEM_PROMPT = """You are FinancialAnalyst.ai, an expert AI financial analyst. 
Your job is to provide thorough, data-driven investment analysis.

When analyzing stocks, always:
1. Gather price data, fundamentals, and technicals
2. Consider both bull and bear cases
3. Identify key risks and catalysts
4. Provide a clear, actionable recommendation

Format your analysis clearly with sections:
- 📊 Current Status (price, key metrics)
- 📈 Technical View (trend, support/resistance)
- 💰 Fundamentals (valuation, growth, margins)
- 📰 Recent News (if relevant)
- ⚠️ Risks
- 🎯 Recommendation

Be honest about uncertainty. If data is limited, say so.
Never guarantee returns or make promises about future performance.

You have access to real-time market data through your tools. Use them."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize the analyst agent."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = FinancialTools()

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        tool_map = {
            "get_stock_price": self.tools.get_stock_price,
            "get_company_info": self.tools.get_company_info,
            "get_technical_analysis": self.tools.get_technical_analysis,
            "get_news": self.tools.get_news,
            "compare_stocks": self.tools.compare_stocks,
        }

        if name not in tool_map:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = tool_map[name](**arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def ask(self, question: str) -> str:
        """Ask the agent a question."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        # Initial response (may include tool calls)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        # Handle tool calls iteratively
        while assistant_message.tool_calls:
            messages.append(assistant_message)

            # Execute all tool calls
            for tool_call in assistant_message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto"
            )
            assistant_message = response.choices[0].message

        return assistant_message.content

    def analyze(self, symbol: str) -> str:
        """Perform a comprehensive analysis of a stock."""
        prompt = f"""Perform a comprehensive analysis of {symbol.upper()}.

Include:
1. Current price and recent performance
2. Key fundamental metrics (PE, margins, growth)
3. Technical analysis (trend, RSI, support/resistance)
4. Recent news if significant
5. Bull case and bear case
6. Your recommendation with confidence level

Be thorough but concise."""

        return self.ask(prompt)

    def review_portfolio(self, positions: list[dict]) -> str:
        """Review a portfolio."""
        portfolio_str = json.dumps(positions, indent=2)
        prompt = f"""Review this portfolio and provide analysis:

{portfolio_str}

Include:
1. Current value and total P&L
2. Sector/concentration analysis
3. Risk assessment
4. Any stocks that need attention
5. Rebalancing suggestions if appropriate"""

        return self.ask(prompt)


# CLI Interface
@click.group()
def cli():
    """FinancialAnalyst.ai - AI-powered financial analysis"""
    pass


@cli.command()
def chat():
    """Start interactive chat session."""
    console.print(Panel.fit(
        "[bold blue]FinancialAnalyst.ai[/bold blue]\n"
        "AI-powered financial analysis\n\n"
        "Type your questions, 'quit' to exit",
        border_style="blue"
    ))

    try:
        agent = FinancialAnalyst()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    while True:
        try:
            question = console.input("\n[bold green]You:[/bold green] ")
            if question.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            with console.status("[bold blue]Analyzing...[/bold blue]"):
                response = agent.ask(question)

            console.print(f"\n[bold blue]Analyst:[/bold blue]")
            console.print(Markdown(response))

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break


@cli.command()
@click.argument("symbol")
def analyze(symbol):
    """Analyze a specific stock."""
    try:
        agent = FinancialAnalyst()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print(f"\n[bold]Analyzing {symbol.upper()}...[/bold]\n")

    with console.status("[bold blue]Fetching data and analyzing...[/bold blue]"):
        result = agent.analyze(symbol)

    console.print(Markdown(result))


@cli.command()
@click.argument("symbols", nargs=-1)
def compare(symbols):
    """Compare multiple stocks."""
    if len(symbols) < 2:
        console.print("[red]Please provide at least 2 symbols to compare[/red]")
        return

    try:
        agent = FinancialAnalyst()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    symbols_str = ", ".join(s.upper() for s in symbols)
    console.print(f"\n[bold]Comparing {symbols_str}...[/bold]\n")

    with console.status("[bold blue]Analyzing...[/bold blue]"):
        result = agent.ask(f"Compare these stocks for investment: {symbols_str}")

    console.print(Markdown(result))


@cli.command()
@click.argument("symbol")
def quick(symbol):
    """Quick price check (no AI, just data)."""
    console.print(f"\n[bold]Quick look: {symbol.upper()}[/bold]\n")

    tools = FinancialTools()
    price_data = tools.get_stock_price(symbol, include_history=True)
    tech_data = tools.get_technical_analysis(symbol)

    # Price table
    table = Table(title=f"{price_data.get('name', symbol)} ({symbol.upper()})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Price", f"${price_data.get('price', 'N/A')}")
    table.add_row("Change", f"{price_data.get('change_percent', 'N/A')}%")
    table.add_row("Day Range", f"${price_data.get('day_low', 'N/A')} - ${price_data.get('day_high', 'N/A')}")
    table.add_row("52W Range", f"${price_data.get('52_week_low', 'N/A')} - ${price_data.get('52_week_high', 'N/A')}")
    table.add_row("P/E Ratio", f"{price_data.get('pe_ratio', 'N/A')}")
    table.add_row("Market Cap", f"${price_data.get('market_cap', 0)/1e9:.1f}B" if price_data.get('market_cap') else "N/A")
    table.add_row("---", "---")
    table.add_row("Trend", tech_data.get("trend", "N/A"))
    table.add_row("RSI", f"{tech_data.get('momentum', {}).get('rsi_14', 'N/A')}")
    table.add_row("Support", f"${tech_data.get('levels', {}).get('support', 'N/A')}")
    table.add_row("Resistance", f"${tech_data.get('levels', {}).get('resistance', 'N/A')}")

    console.print(table)


if __name__ == "__main__":
    cli()
