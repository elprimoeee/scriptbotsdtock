# RSI Bot - Quick Start Guide

## 📊 What This Bot Does

A sophisticated ASX stock trading bot that:
- ✅ Analyzes **top 200 low-volatility stocks** to reduce risk
- ✅ Uses **RSI momentum indicators** to identify buying/selling opportunities
- ✅ Creates **personalized trading ranges** based on each stock's annual performance
- ✅ **Prevents whipsaw losses** by requiring price confirmation before trading
- ✅ **Backtests** complete trading scenarios
- ✅ **Visualizes** performance with detailed charts

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data
Make sure you have the ASX dataset:
```bash
python scripts/build_asx_dataset.py --verbose
```

### Step 3: Run the Bot
```bash
python main.py
```

This runs a complete backtest and generates:
- Equity curve chart
- Per-stock range analysis with buy/sell signals
- Detailed transaction logs
- Summary metrics

## 📈 Daily Usage

### Update Daily Signals
Every day, run:
```bash
python scripts/update_rsi_bot_daily.py
```

This shows:
- Today's recommended **BUY** stocks
- Today's recommended **SELL** stocks
- Saves signals to `results/rsi_bot/daily/`

### Analyze Specific Stocks
```bash
python scripts/plot_stock_analysis.py CBA BHP NAB
```

Creates detailed charts for:
- Price action
- RSI indicator
- Volatility trends
- Momentum vs trading ranges

### Analyze and backtest one stock
```bash
python scripts/run_single_stock.py TLS.AX 5y
```

The HTML report models lagged next-open fills, slippage, brokerage, exposure
limits, cooldowns, and immediate lagged risk exits. It compares the bot with
buy-and-hold and the ASX 200 and reports risk and trade-quality metrics.

### Run the long-term strategy
```bash
python scripts/run_single_stock.py TLS.AX 10y long-term
```

The long-term mode is designed for multi-year ownership. It buys established
uptrends when the 50-day average is above a rising 200-day average, annual
momentum is positive, and the stock has not suffered an excessive drawdown.
It waits at least 126 trading days before an ordinary exit, uses a 63-day buy
cooldown, and sells after a confirmed long-term trend breakdown. A severe
drop below the 200-day average remains an immediate risk exit. Its report is
saved separately as `results/single_stock/TLS.AX_long_term_report.html`.

### Run walk-forward robustness checks
```bash
python scripts/walk_forward_single_stock.py TLS.AX 10y
```

This tunes nearby RSI values on rolling three-year training windows and
evaluates them on unseen six-month windows. Outputs are saved in
`results/robustness/`.

## 📊 Understanding the Output

### Key Files Generated

1. **equity_curve.png** - Shows portfolio growth over time
2. **sample_stocks_ranges.png** - Shows how trading ranges work
3. **positions.csv** - Complete list of all trades
4. **daily_pnl.csv** - Daily portfolio values

### Reading the Charts

**Sample Stock Chart Legend:**
- 🔵 **Blue line** = 3-month momentum (how much stock has moved)
- 🟢 **Green band** = Trading range (buy below, sell above)
- 🔼 **Green triangles** = BUY signals (price rising back into range)
- 🔽 **Red triangles** = SELL signals (price falling back into range)

**Why the confirmation logic matters:**
```
Example: Stock XYZ drops below range
   Day 1: Price below range → DON'T BUY YET (might keep falling)
   Day 2: Price starts rising back up → BUY NOW (reversal confirmed)
```

## 🎯 Key Concepts

### Dynamic Trading Range
For each stock, the range uses the trailing 70th percentile of absolute
three-month momentum, clipped between 3% and 30%. This causal range adapts to
recent behavior without widening merely because the previous one-year return
was strongly positive or negative.

### RSI Momentum
- **RSI = 0-30** = Oversold (potential buy)
- **RSI = 30-70** = Neutral (watching)
- **RSI = 70-100** = Overbought (potential sell)
- Bot favors RSI near 50 (neutral momentum)

### Buy/Sell Confirmation
```
BUY Flow:
1. Stock drops BELOW range (oversold signal)
2. Price starts RISING back up (reversal)
3. EXECUTE BUY when momentum confirms

SELL Flow:
1. Stock rises ABOVE range (overbought signal)
2. Price starts FALLING back down (reversal)
3. EXECUTE SELL when momentum confirms
```

After the minimum holding period, the momentum backtest also uses a
**profit-first exit**. It may sell outside the RSI/range lines once the lot is
profitable after entry and exit commission. If that exit is unavailable, the
original reversal and risk-break rules remain active so a falling stock is not
held indefinitely merely to protect the reported win rate.

## 🔧 Customization

Edit `models/rsi_momentum_bot.py` to adjust:

```python
# Use tighter ranges for more aggressive trading
RANGE_MULTIPLIER = 0.3  # Default 0.5

# Focus on even safer stocks
TOP_STOCKS = 100  # Default 200

# Longer lookback period for different momentum
MOMENTUM_LOOKBACK_DAYS = 126  # Default 63 (~3 months)

# Larger positions
POSITION_SIZE_PCT = 0.05  # Default 0.02 (5% vs 2%)
```

## 📊 Example Daily Report

```
==============================================================================
RSI BOT - DAILY REPORT (2024-05-09)
==============================================================================

BUY SIGNALS: 8 stocks
────────────────────────────────────────────────────────────────────────────
Ticker   RSI      3M Mom     1Y Return   Range               Price     
────────────────────────────────────────────────────────────────────────────
CBA      42.5     -3.50%     +15.20%     -7.60% / +7.60%    $125.50
BHP      38.2     -5.10%     +18.00%     -9.00% / +9.00%    $48.25
... and 6 more

SELL SIGNALS: 3 stocks
────────────────────────────────────────────────────────────────────────────
Ticker   RSI      3M Mom     1Y Return   Range               Price     
────────────────────────────────────────────────────────────────────────────
WES      72.8     +8.20%     +12.50%     -6.25% / +6.25%    $35.80
NCM      68.5     +6.30%     +14.00%     -7.00% / +7.00%    $8.95

Total Tracked Stocks: 200
==============================================================================
```

## ⚠️ Important Notes

1. **Backtesting Tool** - This is NOT connected to live trading
2. **Paper Trading Only** - All results are simulated
3. **Risk Management** - Position sizing limits concentration
4. **Past Performance** - Historical results don't guarantee future returns
5. **Monitor Regularly** - Review strategy performance monthly

## 🆘 Troubleshooting

### "Ticker not found in dataset"
- Run `python scripts/build_asx_dataset.py` to rebuild data
- Check ticker symbol is correct (use uppercase)

### No buy/sell signals
- May be normal if market is neutral
- Check individual stocks with `plot_stock_analysis.py`
- Reduce `RANGE_MULTIPLIER` for tighter ranges

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## 📚 Files Reference

```
main.py
├── Model entry point
├── Change SELECTED_MODEL = "rsi_bot" to run bot

models/rsi_momentum_bot.py
├── Main strategy implementation
├── RSIBuySignal class - calculates indicators
├── execute_rsi_bot_model() - runs backtest

scripts/update_rsi_bot_daily.py
├── Daily update utility
├── Shows today's signals
├── Exports CSV reports

scripts/plot_stock_analysis.py
├── Detailed stock analysis
├── Creates comprehensive charts
├── Usage: python scripts/plot_stock_analysis.py CBA BHP NAB

results/rsi_bot/
├── equity_curve.png
├── sample_stocks_ranges.png
├── positions.csv
├── daily_pnl.csv
└── daily/
    └── signals_YYYY-MM-DD.csv
```

## 💡 Next Steps

1. ✅ Run `python main.py` for first backtest
2. ✅ Review `results/rsi_bot/equity_curve.png`
3. ✅ Analyze top stocks: `python scripts/plot_stock_analysis.py CBA BHP NAB`
4. ✅ Set up daily update: `python scripts/update_rsi_bot_daily.py`
5. ✅ Experiment with parameters for your risk tolerance

## 🤝 Support

For issues or questions:
1. Check RSI_BOT_README.md for detailed documentation
2. Review inline code comments in models/rsi_momentum_bot.py
3. Test with `plot_stock_analysis.py` on specific stocks
