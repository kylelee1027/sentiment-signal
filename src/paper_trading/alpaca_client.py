"""
Alpaca Paper Trading Client

Handles all interactions with the Alpaca API for paper trading.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlpacaClient:
    """Client for interacting with Alpaca Paper Trading API"""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Alpaca client

        Args:
            api_key: Alpaca API key (defaults to APCA-API-KEY-ID from .env)
            secret_key: Alpaca secret key (defaults to APCA-API-SECRET-KEY from .env)
        """
        self.api_key = api_key or os.getenv('APCA-API-KEY-ID')
        self.secret_key = secret_key or os.getenv('APCA-API-SECRET-KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Please set APCA-API-KEY-ID and APCA-API-SECRET-KEY in .env file"
            )

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
            from alpaca.data.requests import StockLatestQuoteRequest

            self.TradingClient = TradingClient
            self.StockHistoricalDataClient = StockHistoricalDataClient
            self.MarketOrderRequest = MarketOrderRequest
            self.GetOrdersRequest = GetOrdersRequest
            self.OrderSide = OrderSide
            self.TimeInForce = TimeInForce
            self.QueryOrderStatus = QueryOrderStatus
            self.StockLatestQuoteRequest = StockLatestQuoteRequest

            # Initialize clients
            self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
            self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

            logger.info("Alpaca paper trading client initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import alpaca-py: {e}")
            logger.error("Install with: pip install alpaca-py")
            raise

    def get_account(self) -> Dict:
        """
        Get account information

        Returns:
            Dictionary with account details (equity, cash, buying_power, etc.)
        """
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'position_market_value': float(account.position_market_value) if account.position_market_value else 0.0,
                'account_blocked': account.account_blocked,
                'trading_blocked': account.trading_blocked
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_positions(self) -> List[Dict]:
        """
        Get all open positions

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for specific symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Position dictionary or None if no position exists
        """
        try:
            pos = self.trading_client.get_open_position(symbol)
            return {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': pos.side,
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price),
                'avg_entry_price': float(pos.avg_entry_price)
            }
        except Exception as e:
            # No position exists
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current ask price or None if not available
        """
        try:
            request = self.StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                # Use ask price for buying, bid for selling
                return float(quote.ask_price)
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def place_market_order(self, symbol: str, qty: float, side: str) -> Optional[Dict]:
        """
        Place a market order

        Args:
            symbol: Stock ticker symbol
            qty: Quantity to trade (can be fractional)
            side: 'buy' or 'sell'

        Returns:
            Order details dictionary or None if failed
        """
        try:
            order_side = self.OrderSide.BUY if side.lower() == 'buy' else self.OrderSide.SELL

            order_request = self.MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=self.TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data=order_request)

            logger.info(f"Order placed: {side.upper()} {qty} shares of {symbol}")

            return {
                'id': str(order.id),
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'type': order.type.value,
                'status': order.status.value,
                'submitted_at': order.submitted_at
            }

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """
        Close an entire position

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Closed position for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False

    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict]:
        """
        Get orders history

        Args:
            status: Order status filter ('open', 'closed', 'all')
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        try:
            if status == 'open':
                query_status = self.QueryOrderStatus.OPEN
            elif status == 'closed':
                query_status = self.QueryOrderStatus.CLOSED
            else:
                query_status = self.QueryOrderStatus.ALL

            request = self.GetOrdersRequest(
                status=query_status,
                limit=limit
            )

            orders = self.trading_client.get_orders(filter=request)

            return [
                {
                    'id': str(order.id),
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                    'side': order.side.value,
                    'type': order.type.value,
                    'status': order.status.value,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at
                }
                for order in orders
            ]

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def calculate_position_size(self, symbol: str, portfolio_pct: float = 0.1) -> Tuple[float, float]:
        """
        Calculate position size based on available buying power

        Args:
            symbol: Stock ticker symbol
            portfolio_pct: Percentage of portfolio to allocate (default 10%)

        Returns:
            Tuple of (quantity, estimated_value)
        """
        try:
            account = self.get_account()
            buying_power = account['buying_power']

            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get price for {symbol}")
                return 0.0, 0.0

            # Calculate position size
            target_value = buying_power * portfolio_pct
            qty = target_value / current_price

            # Round to 2 decimal places for fractional shares
            qty = round(qty, 2)

            estimated_value = qty * current_price

            logger.info(f"Position size for {symbol}: {qty} shares (~${estimated_value:.2f})")

            return qty, estimated_value

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0, 0.0

    def is_market_open(self) -> bool:
        """
        Check if market is currently open

        Returns:
            True if market is open, False otherwise
        """
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def get_market_status(self) -> Dict:
        """
        Get detailed market status

        Returns:
            Dictionary with market status information
        """
        try:
            clock = self.trading_client.get_clock()
            return {
                'is_open': clock.is_open,
                'timestamp': clock.timestamp,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'is_open': False}
