"""
Web scraping module for collecting news and social media data
"""

from .news_scraper import (
    BaseNewsScraper,
    YahooFinanceScraper,
    GoogleNewsScraper,
    NewsAggregator
)
from .utils import (
    clean_text,
    normalize_ticker,
    generate_article_id,
    is_relevant_article,
    ScrapingStats
)

__all__ = [
    'BaseNewsScraper',
    'YahooFinanceScraper',
    'GoogleNewsScraper',
    'NewsAggregator',
    'clean_text',
    'normalize_ticker',
    'generate_article_id',
    'is_relevant_article',
    'ScrapingStats'
]
