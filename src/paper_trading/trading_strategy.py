"""
Paper Trading Strategy Executor

Executes trades based on signals with position management and risk controls.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

from .alpaca_client import AlpacaClient
from .signal_generator import DailySignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTradingStrategy:
    """Executes paper trading strategy with position management"""

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        signal_generator: DailySignalGenerator,
        position_size_pct: float = 0.1,
        max_positions: int = 5,
        log_dir: str = 'logs/paper_trading'
    ):
        """
        Initialize trading strategy

        Args:
            alpaca_client: Alpaca API client
            signal_generator: Signal generator instance
            position_size_pct: Percentage of portfolio per position (default 10%)
            max_positions: Maximum number of concurrent positions
            log_dir: Directory to save trade logs
        """
        self.alpaca = alpaca_client
        self.signal_generator = signal_generator
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        logger.info(
            f"Trading strategy initialized: position_size={position_size_pct*100}%, "
            f"max_positions={max_positions}"
        )

    def execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a trading signal

        Args:
            signal: Signal dictionary from signal generator

        Returns:
            Execution result dictionary
        """
        ticker = signal['ticker']
        signal_type = signal['signal']
        timestamp = datetime.now()

        result = {
            'ticker': ticker,
            'signal': signal_type,
            'timestamp': timestamp,
            'executed': False,
            'order': None,
            'reason': ''
        }

        # Check market status
        if not self.alpaca.is_market_open():
            result['reason'] = 'Market is closed'
            logger.warning(f"Market is closed, cannot execute {signal_type} for {ticker}")
            return result

        try:
            # Get current position
            current_position = self.alpaca.get_position(ticker)
            current_qty = float(current_position['qty']) if current_position else 0.0

            # Execute based on signal type
            if signal_type == 'BUY':
                result = self._execute_buy(ticker, signal, current_qty)

            elif signal_type == 'SELL':
                result = self._execute_sell(ticker, signal, current_qty)

            elif signal_type == 'HOLD':
                result['reason'] = 'HOLD signal - no action'
                logger.info(f"HOLD signal for {ticker} - no action taken")

        except Exception as e:
            result['reason'] = f'Error: {str(e)}'
            logger.error(f"Error executing signal for {ticker}: {e}")

        # Log execution
        self._log_execution(signal, result)

        return result

    def _execute_buy(self, ticker: str, signal: Dict, current_qty: float) -> Dict:
        """Execute BUY signal"""
        result = {
            'ticker': ticker,
            'signal': 'BUY',
            'timestamp': datetime.now(),
            'executed': False,
            'order': None,
            'reason': ''
        }

        # Check if we already have a position
        if current_qty > 0:
            result['reason'] = f'Already have position ({current_qty} shares)'
            logger.info(f"Already have {current_qty} shares of {ticker}, skipping BUY")
            return result

        # Check if we're at max positions
        positions = self.alpaca.get_positions()
        if len(positions) >= self.max_positions:
            result['reason'] = f'Max positions reached ({len(positions)}/{self.max_positions})'
            logger.warning(f"Max positions reached, cannot buy {ticker}")
            return result

        # Calculate position size
        qty, estimated_value = self.alpaca.calculate_position_size(ticker, self.position_size_pct)

        if qty <= 0:
            result['reason'] = 'Invalid position size calculated'
            logger.error(f"Invalid position size for {ticker}")
            return result

        # Place order
        logger.info(f"Placing BUY order: {qty} shares of {ticker} (~${estimated_value:.2f})")
        order = self.alpaca.place_market_order(ticker, qty, 'buy')

        if order:
            result['executed'] = True
            result['order'] = order
            result['reason'] = f'Bought {qty} shares at market price'
            logger.info(f"BUY order executed for {ticker}: {qty} shares")
        else:
            result['reason'] = 'Order placement failed'
            logger.error(f"Failed to place BUY order for {ticker}")

        return result

    def _execute_sell(self, ticker: str, signal: Dict, current_qty: float) -> Dict:
        """Execute SELL signal"""
        result = {
            'ticker': ticker,
            'signal': 'SELL',
            'timestamp': datetime.now(),
            'executed': False,
            'order': None,
            'reason': ''
        }

        # Check if we have a position to sell
        if current_qty <= 0:
            result['reason'] = 'No position to sell'
            logger.info(f"No position in {ticker} to sell")
            return result

        # Close entire position
        logger.info(f"Closing position: {current_qty} shares of {ticker}")
        success = self.alpaca.close_position(ticker)

        if success:
            result['executed'] = True
            result['order'] = {'type': 'close_position', 'qty': current_qty}
            result['reason'] = f'Sold {current_qty} shares at market price'
            logger.info(f"SELL order executed for {ticker}: {current_qty} shares")
        else:
            result['reason'] = 'Position close failed'
            logger.error(f"Failed to close position for {ticker}")

        return result

    def _log_execution(self, signal: Dict, result: Dict):
        """Log trade execution to file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'ticker': signal['ticker'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'sentiment_score': signal['sentiment_score'],
            'article_count': signal['article_count'],
            'signal_reason': signal['reason'],
            'executed': result['executed'],
            'execution_reason': result['reason'],
            'order': result.get('order')
        }

        # Save to daily log file
        log_file = os.path.join(
            self.log_dir,
            f"trades_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )

        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")

    def run_daily_strategy(self, tickers: List[Dict[str, str]]) -> Dict:
        """
        Run daily trading strategy for multiple tickers

        Args:
            tickers: List of dicts with 'ticker' and 'company_name' keys

        Returns:
            Summary dictionary with results
        """
        logger.info(f"Running daily strategy for {len(tickers)} tickers")

        # Get account status
        account = self.alpaca.get_account()
        logger.info(
            f"Account status: equity=${account['equity']:.2f}, "
            f"cash=${account['cash']:.2f}, "
            f"buying_power=${account['buying_power']:.2f}"
        )

        # Get current positions
        positions = self.alpaca.get_positions()
        current_positions = {pos['symbol']: pos['qty'] for pos in positions}

        logger.info(f"Current positions: {list(current_positions.keys())}")

        # Generate signals for all tickers
        signals = self.signal_generator.generate_signals_batch(tickers, current_positions)

        # Execute signals
        results = []
        for signal in signals:
            result = self.execute_signal(signal)
            results.append(result)

        # Compile summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'tickers_analyzed': len(tickers),
            'signals_generated': len(signals),
            'buy_signals': sum(1 for s in signals if s['signal'] == 'BUY'),
            'sell_signals': sum(1 for s in signals if s['signal'] == 'SELL'),
            'hold_signals': sum(1 for s in signals if s['signal'] == 'HOLD'),
            'orders_executed': sum(1 for r in results if r['executed']),
            'account_equity': account['equity'],
            'account_cash': account['cash'],
            'positions': current_positions,
            'results': results
        }

        logger.info(
            f"Strategy run complete: {summary['buy_signals']} BUY, "
            f"{summary['sell_signals']} SELL, {summary['hold_signals']} HOLD | "
            f"{summary['orders_executed']} orders executed"
        )

        # Save summary
        self._save_summary(summary)

        return summary

    def _save_summary(self, summary: Dict):
        """Save daily summary to file"""
        summary_file = os.path.join(
            self.log_dir,
            f"summary_{datetime.now().strftime('%Y-%m-%d')}.json"
        )

        try:
            with open(summary_file, 'w') as f:
                json.dumps(summary, indent=2, default=str)
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")

    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio summary

        Returns:
            Dictionary with portfolio metrics
        """
        account = self.alpaca.get_account()
        positions = self.alpaca.get_positions()

        total_unrealized_pl = sum(pos['unrealized_pl'] for pos in positions)
        total_unrealized_plpc = (
            (total_unrealized_pl / account['equity']) * 100
            if account['equity'] > 0 else 0
        )

        return {
            'timestamp': datetime.now().isoformat(),
            'equity': account['equity'],
            'cash': account['cash'],
            'buying_power': account['buying_power'],
            'position_count': len(positions),
            'position_value': account['position_market_value'],
            'unrealized_pl': total_unrealized_pl,
            'unrealized_plpc': total_unrealized_plpc,
            'positions': positions
        }
