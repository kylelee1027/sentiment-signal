"""
Daily Paper Trading Runner

Run this script daily (e.g., at market open) to generate signals and execute trades.
"""

import logging
import argparse
from datetime import datetime
from typing import List, Dict
import json

from .alpaca_client import AlpacaClient
from .signal_generator import DailySignalGenerator
from .trading_strategy import PaperTradingStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_watchlist(file_path: str) -> List[Dict[str, str]]:
    """
    Load watchlist from JSON file

    Expected format:
    [
        {"ticker": "AAPL", "company_name": "Apple Inc"},
        {"ticker": "GOOGL", "company_name": "Alphabet Inc"}
    ]

    Args:
        file_path: Path to watchlist JSON file

    Returns:
        List of ticker dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            watchlist = json.load(f)
        logger.info(f"Loaded {len(watchlist)} tickers from {file_path}")
        return watchlist
    except Exception as e:
        logger.error(f"Error loading watchlist: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Run daily paper trading strategy')

    # Watchlist options
    parser.add_argument(
        '--watchlist',
        help='Path to JSON file with watchlist (list of dicts with ticker and company_name)'
    )
    parser.add_argument(
        '--ticker',
        help='Single ticker to trade (alternative to watchlist)'
    )
    parser.add_argument(
        '--company',
        help='Company name (required if using --ticker)'
    )

    # Signal generation parameters
    parser.add_argument(
        '--model',
        default='finbert',
        choices=['finbert', 'vader', 'textblob'],
        help='Sentiment model to use (default: finbert)'
    )
    parser.add_argument(
        '--buy-threshold',
        type=float,
        default=0.3,
        help='Buy signal threshold (default: 0.3)'
    )
    parser.add_argument(
        '--sell-threshold',
        type=float,
        default=-0.3,
        help='Sell signal threshold (default: -0.3)'
    )
    parser.add_argument(
        '--min-articles',
        type=int,
        default=3,
        help='Minimum articles required for signal (default: 3)'
    )
    parser.add_argument(
        '--lookback-hours',
        type=int,
        default=24,
        help='Hours to look back for news (default: 24)'
    )

    # Trading parameters
    parser.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Position size as percentage of portfolio (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=5,
        help='Maximum concurrent positions (default: 5)'
    )

    # Other options
    parser.add_argument(
        '--log-dir',
        default='logs/paper_trading',
        help='Directory for trade logs (default: logs/paper_trading)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate signals but do not execute trades'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.watchlist and not (args.ticker and args.company):
        parser.error("Must provide either --watchlist or both --ticker and --company")

    # Build ticker list
    if args.watchlist:
        tickers = load_watchlist(args.watchlist)
        if not tickers:
            logger.error("No tickers loaded from watchlist")
            return
    else:
        tickers = [{'ticker': args.ticker, 'company_name': args.company}]

    logger.info(f"Starting daily paper trading run at {datetime.now()}")
    logger.info(f"Tickers to analyze: {[t['ticker'] for t in tickers]}")

    try:
        # Initialize Alpaca client
        logger.info("Initializing Alpaca client...")
        alpaca_client = AlpacaClient()

        # Check market status
        market_status = alpaca_client.get_market_status()
        logger.info(f"Market status: {'OPEN' if market_status['is_open'] else 'CLOSED'}")

        if not market_status['is_open'] and not args.dry_run:
            logger.warning("Market is closed. Signals will be generated but trades cannot be executed.")

        # Initialize signal generator
        logger.info(f"Initializing signal generator with {args.model} model...")
        signal_generator = DailySignalGenerator(
            model_type=args.model,
            sentiment_threshold_buy=args.buy_threshold,
            sentiment_threshold_sell=args.sell_threshold,
            min_articles=args.min_articles,
            lookback_hours=args.lookback_hours
        )

        # Initialize trading strategy
        logger.info("Initializing trading strategy...")
        strategy = PaperTradingStrategy(
            alpaca_client=alpaca_client,
            signal_generator=signal_generator,
            position_size_pct=args.position_size,
            max_positions=args.max_positions,
            log_dir=args.log_dir
        )

        # Get portfolio status
        portfolio = strategy.get_portfolio_summary()
        logger.info(
            f"Portfolio: ${portfolio['equity']:.2f} equity, "
            f"{portfolio['position_count']} positions, "
            f"${portfolio['unrealized_pl']:.2f} unrealized P/L ({portfolio['unrealized_plpc']:.2f}%)"
        )

        if args.dry_run:
            logger.info("DRY RUN MODE - Generating signals only, no trades will be executed")

            # Get current positions
            positions = alpaca_client.get_positions()
            current_positions = {pos['symbol']: pos['qty'] for pos in positions}

            # Generate signals
            signals = signal_generator.generate_signals_batch(tickers, current_positions)

            # Print signals
            print("\n" + "="*80)
            print(f"SIGNALS GENERATED - {datetime.now()}")
            print("="*80)

            for signal in signals:
                print(f"\n{signal['ticker']}:")
                print(f"  Signal: {signal['signal']}")
                print(f"  Confidence: {signal['confidence']:.2f}")
                print(f"  Sentiment: {signal['sentiment_score']:.3f}")
                print(f"  Articles: {signal['article_count']}")
                print(f"  Reason: {signal['reason']}")

            print("\n" + "="*80)

        else:
            # Run strategy
            summary = strategy.run_daily_strategy(tickers)

            # Print summary
            print("\n" + "="*80)
            print(f"DAILY STRATEGY RUN COMPLETE - {datetime.now()}")
            print("="*80)
            print(f"Tickers analyzed: {summary['tickers_analyzed']}")
            print(f"Signals: {summary['buy_signals']} BUY, {summary['sell_signals']} SELL, {summary['hold_signals']} HOLD")
            print(f"Orders executed: {summary['orders_executed']}")
            print(f"Account equity: ${summary['account_equity']:.2f}")
            print(f"Cash available: ${summary['account_cash']:.2f}")
            print(f"Current positions: {list(summary['positions'].keys())}")
            print("="*80 + "\n")

            # Print execution details
            for result in summary['results']:
                if result['executed']:
                    print(f"✓ {result['signal']} {result['ticker']}: {result['reason']}")
                elif result['signal'] != 'HOLD':
                    print(f"✗ {result['signal']} {result['ticker']}: {result['reason']}")

        logger.info("Daily trading run completed successfully")

    except Exception as e:
        logger.error(f"Error during trading run: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
