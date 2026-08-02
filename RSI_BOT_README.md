# RSI Momentum Trading Bot

A sophisticated stock trading bot for the ASX that uses RSI (Relative Strength Index) momentum indicators to identify and trade low-volatility stocks within dynamic price ranges.

## Strategy Overview

### Core Principles

1. **Low Volatility Selection**: Selects the top 200 ASX stocks with the lowest volatility
2. **Dynamic Range Trading**: Creates personalized trading ranges based on each stock's 1-year performance
3. **RSI Momentum**: Uses 14-period RSI to identify momentum shifts (3-month lookback)
4. **Confirmation Logic**: Prevents whipsaw losses by requiring price confirmation before buy/sell execution

### How It Works

#### 1. Stock Selection
- Filters for stocks with minimum $500K daily dollar volume
- Ranks all stocks by volatility (standard deviation of returns)
- Selects the top 200 lowest-volatility stocks
- Rationale: Lower volatility = more predictable price action = fewer false signals

#### 2. Dynamic Range Calculation
For each stock, based on its 1-year performance:
- **Annual Return Calculation**: Returns over the last 252 trading days
- **Range Offset**: Annual return × 0.5 (multiplier)
- **Example**: 
  - Stock gained 10% in the last year
  - Range offset = 10% × 0.5 = 5%
  - Upper band: +5%, Lower band: -5%
  - This becomes the personalized trading range

#### 3. RSI-Based Signals
- **RSI Calculation**: 14-period RSI on 3-month (63-day) momentum
- RSI values: 0-100 (30 = oversold, 70 = overbought, 50 = neutral)
- **Score Generation**: 
  - RSI component (50% weight): Favors neutral RSI (closer to 50)
  - Volatility component (50% weight): Penalizes high volatility

#### 4. Buy/Sell Confirmation Logic

**BUY Signal**:
1. Price drops below the lower band (momentum < -range%)
2. Price starts rising back above the lower band (reversal confirmation)
3. Only then execute the buy

**SELL Signal**:
1. Price rises above the upper band (momentum > +range%)
2. Price starts dropping back below the upper band (reversal confirmation)
3. Only then execute the sell

**Why This Matters**: 
- Prevents buying just before crashes
- Prevents selling right before rallies
- Requires confirmation that the trend is actually reversing

### Configuration Parameters

```python
RSI_PERIOD = 14                  # RSI calculation window
MOMENTUM_LOOKBACK_DAYS = 63      # ~3 months for momentum calculation
ANNUAL_LOOKBACK_DAYS = 252       # ~1 year for range calculation
RANGE_MULTIPLIER = 0.5           # Range = annual_return × 0.5
TOP_STOCKS = 200                 # Number of stocks to trade
VOLATILITY_WINDOW = 20           # Window for volatility calculation
MIN_DOLLAR_VOLUME = 500_000      # Minimum daily dollar volume filter
POSITION_SIZE_PCT = 0.02         # 2% of portfolio per position
```

## Running the Bot

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure you have the ASX dataset built:
```bash
python scripts/build_asx_dataset.py --verbose
```

### 3. Run the Bot
```bash
python main.py
```

Or run directly:
```bash
python models/rsi_momentum_bot.py
```

### 4. Daily Updates
Use the provided daily update script to refresh calculations:
```bash
python scripts/update_rsi_bot_daily.py
```

## Output Files

The bot generates the following output files in `results/rsi_bot/`:

1. **equity_curve.png** - Portfolio value over time
2. **sample_stocks_ranges.png** - Individual stock ranges with buy/sell signals
3. **positions.csv** - All buy/sell transactions with timestamps, prices, and units
4. **daily_pnl.csv** - Daily portfolio values, cash levels, and position counts
5. **summary_metrics** - Performance statistics (return, drawdown, trade count)

## Performance Metrics

The bot tracks:
- **Total Return %**: Overall portfolio performance
- **Max Drawdown %**: Largest peak-to-trough decline
- **Trade Count**: Total number of buy/sell operations
- **Buy/Sell Ratio**: Balance of transactions

## Risk Management

1. **Position Sizing**: 2% of portfolio per position limits concentration risk
2. **Stock Selection**: Top 200 reduces single-stock idiosyncratic risk
3. **Low Volatility Focus**: Inherently less risky than high-volatility picks
4. **Dynamic Ranges**: Personalized for each stock's behavior
5. **Confirmation Logic**: Reduces false signals and whipsaws

## Backtesting

The bot includes a full backtesting engine that:
- Simulates real trading with realistic position sizing
- Tracks all transactions
- Calculates daily portfolio values
- Generates equity curve and other performance metrics

## Customization

To adjust the strategy, modify parameters in `models/rsi_momentum_bot.py`:

- **More aggressive**: Lower `RANGE_MULTIPLIER` (e.g., 0.3) for tighter ranges
- **More conservative**: Raise `RANGE_MULTIPLIER` (e.g., 0.7) for wider ranges
- **Fewer stocks**: Reduce `TOP_STOCKS` to focus on safest names
- **More positions**: Increase `POSITION_SIZE_PCT` for higher leverage
- **Different momentum window**: Adjust `MOMENTUM_LOOKBACK_DAYS`

## Advanced Features (Coming Soon)

- [ ] Stop loss implementation
- [ ] Take profit targets
- [ ] Sector diversification rules
- [ ] Real-time trading integration
- [ ] Email/SMS alerts
- [ ] Web dashboard for monitoring
- [ ] Portfolio rebalancing optimization
- [ ] Machine learning signal enhancement

## Files

- **models/rsi_momentum_bot.py** - Main strategy implementation
- **scripts/update_rsi_bot_daily.py** - Daily update utility
- **results/rsi_bot/** - Output directory for backtests and visualizations

## Notes

- This is a **backtesting and paper trading tool** - not connected to live trading
- Historical performance does not guarantee future results
- ASX market conditions change; monitor strategy performance regularly
- Use appropriate risk management in any real trading implementation
