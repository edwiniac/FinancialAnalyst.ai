# 🤖 FinancialAnalyst.ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous AI agent for financial analysis and investment research. Ask questions in plain English, get detailed analysis with data-backed insights.

## ✨ What It Does

```
You: "Should I buy NVIDIA stock right now?"

Agent: Based on my analysis of NVDA:

📊 Current Status:
- Price: $875.50 (+2.3% today)
- P/E Ratio: 65.2 (elevated vs sector avg of 28)
- RSI: 72.4 (overbought territory)

📈 Technical View:
- Trading above 20/50/200 MA (bullish structure)
- Near resistance at $900
- Support at $820

💰 Fundamentals:
- Revenue growth: +122% YoY (exceptional)
- Gross margin: 74.5% (best in class)
- Data center demand remains strong

⚠️ Risks:
- High valuation assumes continued growth
- China export restrictions
- Competition from AMD, custom chips

🎯 My Take:
Strong company but currently overbought. Consider:
1. Waiting for pullback to $820 support
2. Dollar-cost averaging if long-term bullish
3. Setting stop-loss if entering now
```

## 🚀 Features

- **Natural Language Interface** — Ask questions like you'd ask a financial advisor
- **Multi-Source Analysis** — Combines price data, fundamentals, news, and technicals
- **Autonomous Research** — Agent decides what data to fetch and how to analyze it
- **Portfolio Analysis** — Review holdings, suggest rebalancing, track performance
- **Report Generation** — Create detailed PDF/Markdown reports
- **Watchlist Monitoring** — Get alerts when conditions are met

## 📦 Installation

```bash
git clone https://github.com/edwiniac/FinancialAnalyst.ai.git
cd FinancialAnalyst.ai

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY
```

## 🎮 Usage

### CLI Mode
```bash
# Interactive chat
python analyst.py chat

# One-shot analysis
python analyst.py analyze AAPL

# Portfolio review
python analyst.py portfolio --file my_holdings.json

# Generate report
python analyst.py report TSLA --output report.md
```

### Python API
```python
from analyst import FinancialAnalyst

agent = FinancialAnalyst()

# Ask anything
response = agent.ask("Compare AAPL and MSFT for long-term investment")
print(response)

# Analyze a stock
analysis = agent.analyze("NVDA")
print(analysis.summary)
print(analysis.recommendation)

# Check portfolio
portfolio = [
    {"symbol": "AAPL", "shares": 50, "cost_basis": 150},
    {"symbol": "GOOGL", "shares": 20, "cost_basis": 140},
]
review = agent.review_portfolio(portfolio)
```

### Web UI (Optional)
```bash
python app.py
# Open http://localhost:8000
```

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    User Question                         │
│        "Is Tesla a good investment right now?"          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Planning Agent                         │
│  Decides what information is needed:                    │
│  - Current price & technicals                           │
│  - Financial statements                                 │
│  - Recent news & sentiment                              │
│  - Peer comparison                                      │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Price   │ │Financials│ │  News    │
    │  Tool    │ │  Tool    │ │  Tool    │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────────┴────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Analysis Agent                          │
│  Synthesizes data into actionable insights              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Structured Response                     │
│  Summary → Key Metrics → Analysis → Recommendation      │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Available Tools

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `get_stock_price` | Real-time price and daily stats |
| `get_company_info` | Fundamentals, financials, ratios |
| `get_technical_analysis` | MA, RSI, MACD, support/resistance |
| `get_news` | Recent news and sentiment |
| `get_earnings` | Earnings dates and history |
| `compare_stocks` | Side-by-side comparison |
| `screen_stocks` | Find stocks matching criteria |
| `calculate_portfolio` | Portfolio value and P&L |
| `get_market_overview` | Market indices and status |

## 📋 Example Queries

**Stock Analysis:**
- "Analyze AAPL"
- "Is NVDA overvalued?"
- "What's the technical outlook for TSLA?"
- "Show me META's financials"

**Comparisons:**
- "Compare GOOGL vs MSFT vs AMZN"
- "Which cloud stock has the best fundamentals?"
- "Best dividend stocks in tech sector"

**Portfolio:**
- "Review my portfolio: 100 AAPL @ $150, 50 MSFT @ $350"
- "How should I rebalance for lower risk?"
- "What's my exposure to tech sector?"

**Research:**
- "Find undervalued tech stocks"
- "Which stocks have earnings next week?"
- "Show me stocks with PE < 20 and dividend > 2%"

## 🔧 Configuration

```yaml
# config.yaml
llm:
  provider: openai  # or anthropic
  model: gpt-4o     # or claude-3-opus
  temperature: 0.3

tools:
  data_source: yfinance  # or mcp-finance
  cache_ttl: 300  # seconds

analysis:
  include_news: true
  include_technicals: true
  include_peers: true
```

## 🔌 MCP Integration

This agent can use the [mcp-finance](https://github.com/edwiniac/mcp-finance) server for data:

```yaml
# config.yaml
tools:
  data_source: mcp
  mcp_server: /path/to/mcp-finance/server.py
```

## 📊 Output Formats

- **CLI**: Rich terminal output with tables and colors
- **Markdown**: Clean markdown for reports
- **JSON**: Structured data for integrations
- **PDF**: Professional reports (with `--pdf` flag)

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Always:
- Do your own research
- Consult a licensed financial advisor
- Never invest more than you can afford to lose

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with** [OpenAI](https://openai.com) / [Anthropic](https://anthropic.com) • [yfinance](https://github.com/ranaroussi/yfinance) • Python
