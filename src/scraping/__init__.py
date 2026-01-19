"""
Web scraping module for collecting news and social media data
"""

from .news_scraper import (
    BaseNewsScraper,
    YahooFinanceScraper,
    GoogleNewsScraper,
    NewsAPIScraper,
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
    'NewsAPIScraper',
    'NewsAggregator',
    'clean_text',
    'normalize_ticker',
    'generate_article_id',
    'is_relevant_article',
    'ScrapingStats'
]
