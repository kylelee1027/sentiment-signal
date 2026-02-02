"""
Daily Signal Generator

Generates trading signals based on sentiment analysis of news articles.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..scraping.news_scraper import NewsAggregator
from ..sentiment.analyzer import SentimentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailySignalGenerator:
    """Generates daily trading signals based on news sentiment"""

    def __init__(
        self,
        model_type: str = 'finbert',
        sentiment_threshold_buy: float = 0.3,
        sentiment_threshold_sell: float = -0.3,
        min_articles: int = 3,
        lookback_hours: int = 24
    ):
        """
        Initialize signal generator

        Args:
            model_type: Sentiment model to use ('finbert', 'vader', 'textblob')
            sentiment_threshold_buy: Minimum compound score to generate BUY signal
            sentiment_threshold_sell: Maximum compound score to generate SELL signal
            min_articles: Minimum number of articles required to generate signal
            lookback_hours: Number of hours to look back for news (default 24)
        """
        self.model_type = model_type
        self.sentiment_threshold_buy = sentiment_threshold_buy
        self.sentiment_threshold_sell = sentiment_threshold_sell
        self.min_articles = min_articles
        self.lookback_hours = lookback_hours

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentPipeline(model_type=model_type)

        # Initialize news aggregator
        self.news_aggregator = NewsAggregator()

        logger.info(
            f"Signal generator initialized: model={model_type}, "
            f"buy_threshold={sentiment_threshold_buy}, "
            f"sell_threshold={sentiment_threshold_sell}"
        )

    def collect_recent_news(self, ticker: str, company_name: str) -> pd.DataFrame:
        """
        Collect recent news for a ticker

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search

        Returns:
            DataFrame with recent news articles
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=self.lookback_hours)

        logger.info(f"Collecting news for {ticker} from last {self.lookback_hours} hours")

        # Collect news from all sources
        df = self.news_aggregator.collect_news(
            ticker=ticker,
            company_name=company_name,
            start_date=start_date,
            end_date=end_date
        )

        return df

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles

        Args:
            df: DataFrame with news articles

        Returns:
            DataFrame with sentiment scores added
        """
        if df.empty:
            logger.warning("No articles to analyze")
            return df

        logger.info(f"Analyzing sentiment for {len(df)} articles")

        # Analyze sentiment
        df = self.sentiment_analyzer.analyze_articles(df)

        return df

    def generate_signal(
        self,
        ticker: str,
        company_name: str,
        current_position: Optional[float] = None
    ) -> Dict:
        """
        Generate trading signal for a ticker

        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            current_position: Current position quantity (None if no position)

        Returns:
            Dictionary with signal details:
            {
                'ticker': str,
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': float (0-1),
                'sentiment_score': float (-1 to 1),
                'article_count': int,
                'timestamp': datetime,
                'reason': str,
                'articles': DataFrame
            }
        """
        timestamp = datetime.now()

        # Collect recent news
        news_df = self.collect_recent_news(ticker, company_name)

        # Check if we have enough articles
        if len(news_df) < self.min_articles:
            logger.warning(
                f"Insufficient articles for {ticker}: {len(news_df)} < {self.min_articles}"
            )
            return {
                'ticker': ticker,
                'signal': 'HOLD',
                'confidence': 0.0,
                'sentiment_score': 0.0,
                'article_count': len(news_df),
                'timestamp': timestamp,
                'reason': f'Insufficient articles ({len(news_df)} < {self.min_articles})',
                'articles': news_df
            }

        # Analyze sentiment
        news_df = self.analyze_sentiment(news_df)

        # Calculate aggregate sentiment metrics
        avg_sentiment = news_df['sentiment_compound'].mean()
        sentiment_std = news_df['sentiment_compound'].std()
        positive_ratio = (news_df['sentiment_label'] == 'positive').sum() / len(news_df)
        negative_ratio = (news_df['sentiment_label'] == 'negative').sum() / len(news_df)

        logger.info(
            f"{ticker} sentiment: avg={avg_sentiment:.3f}, std={sentiment_std:.3f}, "
            f"pos_ratio={positive_ratio:.2f}, neg_ratio={negative_ratio:.2f}"
        )

        # Generate signal
        signal = 'HOLD'
        confidence = 0.0
        reason = ''

        # BUY signal logic
        if avg_sentiment >= self.sentiment_threshold_buy:
            if current_position is None or current_position == 0:
                signal = 'BUY'
                confidence = min(abs(avg_sentiment), 1.0)
                reason = f'Positive sentiment ({avg_sentiment:.3f} >= {self.sentiment_threshold_buy})'
            else:
                signal = 'HOLD'
                confidence = min(abs(avg_sentiment), 1.0)
                reason = f'Already have position, positive sentiment maintained ({avg_sentiment:.3f})'

        # SELL signal logic
        elif avg_sentiment <= self.sentiment_threshold_sell:
            if current_position and current_position > 0:
                signal = 'SELL'
                confidence = min(abs(avg_sentiment), 1.0)
                reason = f'Negative sentiment ({avg_sentiment:.3f} <= {self.sentiment_threshold_sell})'
            else:
                signal = 'HOLD'
                confidence = 0.0
                reason = f'No position to sell, negative sentiment ({avg_sentiment:.3f})'

        # HOLD logic
        else:
            signal = 'HOLD'
            confidence = 0.0
            reason = f'Neutral sentiment ({avg_sentiment:.3f})'

        logger.info(f"{ticker} signal: {signal} (confidence: {confidence:.2f}) - {reason}")

        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'sentiment_score': avg_sentiment,
            'sentiment_std': sentiment_std,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'article_count': len(news_df),
            'timestamp': timestamp,
            'reason': reason,
            'articles': news_df
        }

    def generate_signals_batch(
        self,
        tickers: List[Dict[str, str]],
        current_positions: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Generate signals for multiple tickers

        Args:
            tickers: List of dicts with 'ticker' and 'company_name' keys
            current_positions: Dict mapping ticker -> current quantity

        Returns:
            List of signal dictionaries
        """
        if current_positions is None:
            current_positions = {}

        signals = []
        for ticker_info in tickers:
            ticker = ticker_info['ticker']
            company_name = ticker_info['company_name']
            current_pos = current_positions.get(ticker)

            try:
                signal = self.generate_signal(ticker, company_name, current_pos)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {ticker}: {e}")
                signals.append({
                    'ticker': ticker,
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'sentiment_score': 0.0,
                    'article_count': 0,
                    'timestamp': datetime.now(),
                    'reason': f'Error: {str(e)}',
                    'articles': pd.DataFrame()
                })

        return signals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate daily trading signals')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    parser.add_argument('--company', required=True, help='Company name')
    parser.add_argument('--model', default='finbert', choices=['finbert', 'vader', 'textblob'],
                       help='Sentiment model (default: finbert)')
    parser.add_argument('--buy-threshold', type=float, default=0.3,
                       help='Buy signal threshold (default: 0.3)')
    parser.add_argument('--sell-threshold', type=float, default=-0.3,
                       help='Sell signal threshold (default: -0.3)')
    parser.add_argument('--min-articles', type=int, default=3,
                       help='Minimum articles required (default: 3)')

    args = parser.parse_args()

    # Initialize generator
    generator = DailySignalGenerator(
        model_type=args.model,
        sentiment_threshold_buy=args.buy_threshold,
        sentiment_threshold_sell=args.sell_threshold,
        min_articles=args.min_articles
    )

    # Generate signal
    signal = generator.generate_signal(args.ticker, args.company)

    # Print results
    print(f"\n{'='*60}")
    print(f"SIGNAL GENERATED: {signal['signal']}")
    print(f"{'='*60}")
    print(f"Ticker: {signal['ticker']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"Sentiment Score: {signal['sentiment_score']:.3f}")
    print(f"Article Count: {signal['article_count']}")
    print(f"Timestamp: {signal['timestamp']}")
    print(f"Reason: {signal['reason']}")
    print(f"{'='*60}\n")

    if not signal['articles'].empty:
        print("Recent Articles:")
        print(signal['articles'][['date', 'source', 'title', 'sentiment_label', 'sentiment_compound']].to_string())
