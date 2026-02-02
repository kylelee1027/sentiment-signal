# Paper Trading Module

Live paper trading with daily sentiment-based signals using Alpaca API.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Alpaca API Keys

Add your Alpaca paper trading API keys to the `.env` file:

```bash
# .env file
APCA-API-KEY-ID=your_api_key_id_here
APCA-API-SECRET-KEY=your_secret_key_here
NEWS_API_KEY=your_existing_news_api_key
```

To get your Alpaca API keys:
1. Go to https://app.alpaca.markets/login
2. Switch to **Paper Trading** account (upper-left corner)
3. Navigate to the API section
4. Generate new API keys
5. Copy both the Key ID and Secret Key to your `.env` file

### 3. Create Watchlist

Create a `watchlist.json` file with the stocks you want to trade:

```json
[
  {
    "ticker": "AAPL",
    "company_name": "Apple Inc"
  },
  {
    "ticker": "GOOGL",
    "company_name": "Alphabet Inc"
  }
]
```

A sample watchlist is included in the project root.

## Usage

### Run Daily Strategy

Execute the daily trading strategy at market open (recommended: 9:30 AM ET):

```bash
# Using watchlist
python -m src.paper_trading.run_daily --watchlist watchlist.json

# Single ticker
python -m src.paper_trading.run_daily --ticker AAPL --company "Apple Inc"
```

### Dry Run (Test Without Trading)

Generate signals without executing trades:

```bash
python -m src.paper_trading.run_daily \
  --watchlist watchlist.json \
  --dry-run
```

### Customize Parameters

```bash
python -m src.paper_trading.run_daily \
  --watchlist watchlist.json \
  --model finbert \
  --buy-threshold 0.3 \
  --sell-threshold -0.3 \
  --min-articles 3 \
  --lookback-hours 24 \
  --position-size 0.1 \
  --max-positions 5
```

**Parameters:**
- `--model`: Sentiment model (`finbert`, `vader`, `textblob`)
- `--buy-threshold`: Minimum sentiment score to trigger BUY (default: 0.3)
- `--sell-threshold`: Maximum sentiment score to trigger SELL (default: -0.3)
- `--min-articles`: Minimum articles required for signal (default: 3)
- `--lookback-hours`: Hours to look back for news (default: 24)
- `--position-size`: Position size as % of portfolio (default: 0.1 = 10%)
- `--max-positions`: Maximum concurrent positions (default: 5)

## How It Works

### Signal Generation

1. **Collect News**: Scrapes recent news (last 24 hours by default) from multiple sources
2. **Analyze Sentiment**: Uses FinBERT (or VADER/TextBlob) to analyze article sentiment
3. **Generate Signal**: Calculates aggregate sentiment score and generates trading signal

**Signal Logic:**
- **BUY**: Sentiment ≥ 0.3 AND no current position
- **SELL**: Sentiment ≤ -0.3 AND have current position
- **HOLD**: Sentiment between -0.3 and 0.3 OR already at target position

### Trade Execution

1. **Check Market**: Verify market is open
2. **Position Management**:
   - Max 5 concurrent positions (configurable)
   - Each position = 10% of buying power (configurable)
3. **Execute Orders**: Place market orders via Alpaca API
4. **Log Results**: Save trade logs and daily summary

### Logs

Trade logs are saved to `logs/paper_trading/`:
- `trades_YYYY-MM-DD.jsonl`: Individual trade executions
- `summary_YYYY-MM-DD.json`: Daily summary with all signals and results

## Strategy Details

### Position Sizing
- Default: 10% of buying power per position
- Fractional shares supported
- Example: $10,000 buying power → $1,000 per position

### Risk Management
- Maximum positions limit (default: 5)
- No position will be opened if at max positions
- Entire position closed on SELL signal
- Only trades during market hours

### News Sources
- Yahoo Finance (via yfinance API)
- Google News
- NewsAPI.org

## Monitoring Your Trades

### Check Account Status

```python
from src.paper_trading.alpaca_client import AlpacaClient

client = AlpacaClient()

# Get account info
account = client.get_account()
print(f"Equity: ${account['equity']:.2f}")
print(f"Cash: ${account['cash']:.2f}")

# Get positions
positions = client.get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['qty']} shares, P/L: ${pos['unrealized_pl']:.2f}")
```

### View in Alpaca Dashboard

1. Go to https://app.alpaca.markets/paper/dashboard
2. View positions, orders, and account history

## Automation

### Schedule Daily Runs

Use cron (Linux/Mac) or Task Scheduler (Windows) to run daily at market open:

**Cron example (9:30 AM ET on weekdays):**
```bash
30 9 * * 1-5 cd /path/to/sentiment-signal && /path/to/python -m src.paper_trading.run_daily --watchlist watchlist.json
```

## Testing Workflow

1. **Day 1**: Run with `--dry-run` to verify signals
2. **Day 2-7**: Run live with small position sizes (e.g., `--position-size 0.05`)
3. **Week 2+**: Scale up position sizes as confidence grows

## Troubleshooting

### "Alpaca API credentials not found"
- Verify `.env` file exists and has correct keys
- Check key names: `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`

### "Market is closed"
- Alpaca paper trading follows real market hours (9:30 AM - 4:00 PM ET, weekdays)
- Use `--dry-run` to test signals outside market hours

### "Insufficient articles"
- Lower `--min-articles` parameter
- Check if news scrapers are working (`python -m src.scraping.news_scraper --ticker AAPL --company "Apple Inc" --start-date 2024-01-01 --end-date 2024-01-02`)

## Example Output

```
================================================================================
DAILY STRATEGY RUN COMPLETE - 2024-01-19 09:35:00
================================================================================
Tickers analyzed: 5
Signals: 2 BUY, 1 SELL, 2 HOLD
Orders executed: 3
Account equity: $100500.00
Cash available: $89500.00
Current positions: ['AAPL', 'GOOGL']
================================================================================

✓ BUY AAPL: Bought 50 shares at market price
✓ BUY GOOGL: Bought 75 shares at market price
✓ SELL TSLA: Sold 100 shares at market price
```

## Next Steps

- Monitor performance for 1-2 weeks
- Adjust thresholds based on results
- Consider adding more sophisticated risk management
- Implement stop-loss orders
- Add backtesting with historical data
