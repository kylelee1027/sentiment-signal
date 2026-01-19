"""
News Scraper for Stock Sentiment Analysis

Scrapes news articles from multiple sources (Yahoo Finance, Google News, etc.)
for a given company/ticker and date range.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging
from urllib.parse import quote
import re
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseNewsScraper:
    """Base class for news scrapers"""

    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Args:
            rate_limit_delay: Delay between requests in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def scrape(self, ticker: str, company_name: str,
               start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Abstract method to be implemented by subclasses

        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            start_date: Start date for articles
            end_date: End date for articles

        Returns:
            List of article dictionaries
        """
        raise NotImplementedError


class YahooFinanceScraper(BaseNewsScraper):
    """Scraper for Yahoo Finance news using yfinance API"""

    def __init__(self, rate_limit_delay: float = 1.0):
        super().__init__(rate_limit_delay)

    def scrape(self, ticker: str) -> List[Dict]:
        """
        Scrape news from Yahoo Finance using yfinance API. Note that this will be only used to gather most recent news articles

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            List of dictionaries containing article data
        """
        articles = []

        logger.info(f"Fetching Yahoo Finance news for {ticker} via yfinance API")

        try:
            time.sleep(self.rate_limit_delay)

            # Use Search class to get news
            search = yf.Search(ticker, max_results=50)

            if hasattr(search, 'news') and search.news:
                logger.info(f"Found {len(search.news)} news items from Search API")
                for item in search.news:
                    try:
                        # Extract timestamp and convert to datetime
                        timestamp = item.get('providerPublishTime', 0)
                        
                        article_date = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()

                        # Extract article fields
                        title = item.get('title', '')
                        link = item.get('link', '')
                        publisher = item.get('publisher', 'Yahoo Finance')
                        article_type = item.get('type', 'STORY')
                        uuid = item.get('uuid', '')

                        # Extract thumbnail URL if available
                        thumbnail_url = ''
                        thumbnail = item.get('thumbnail', {})
                        if isinstance(thumbnail, dict) and 'resolutions' in thumbnail:
                            resolutions = thumbnail['resolutions']
                            if resolutions and len(resolutions) > 0:
                                # Get the first (original) resolution
                                thumbnail_url = resolutions[0].get('url', '')

                        # Note: The Search API doesn't provide article content/summary
                        # Content would need to be scraped from the article URL separately
                        articles.append({
                            'title': title,
                            'content': '',  # Not available in Search API
                            'url': link,
                            'source': publisher,
                            'date': article_date,
                            'ticker': ticker,
                            'uuid': uuid,
                            'type': article_type,
                            'thumbnail_url': thumbnail_url
                        })
                        print(articles)

                    except Exception as e:
                        logger.debug(f"Error parsing news item: {e}")
                        continue

            logger.info(f"Collected {len(articles)} articles from Yahoo Finance API")

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")

        return articles


class GoogleNewsScraper(BaseNewsScraper):
    """Scraper for Google News via search"""

    def __init__(self, rate_limit_delay: float = 2.0):
        super().__init__(rate_limit_delay)
        self.base_url = "https://www.google.com/search"

    def scrape(self, ticker: str, company_name: str,
               start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Scrape news from Google News search

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search query
            start_date: Start date for news articles
            end_date: End date for news articles

        Returns:
            List of dictionaries containing article data
        """
        articles = []

        # Build search query
        query = f"{company_name} stock news"

        logger.info(f"Scraping Google News for {company_name}")

        try:
            url = f"{self.base_url}?q={quote(query)}&tbm=nws"
            response = self._make_request(url)

            if not response:
                return articles

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find news articles
            article_elements = soup.find_all('div', class_='SoaBEf')

            if not article_elements:
                # Alternative selector
                article_elements = soup.find_all('div', {'data-sokoban-container': True})

            for article in article_elements[:30]:
                try:
                    # Extract title and link
                    title_elem = article.find('div', role='heading') or article.find('h3')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    link_elem = article.find('a')
                    link = link_elem['href'] if link_elem else ""

                    # Extract source
                    source_elem = article.find('div', class_='CEMjEf')
                    source = source_elem.get_text(strip=True) if source_elem else "Google News"

                    # Extract snippet
                    snippet_elem = article.find('div', class_='GI74Re')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    # Extract date
                    date_elem = article.find('span', class_='r0bn4c')
                    article_date = self._parse_date(date_elem.get_text(strip=True)) if date_elem else datetime.now()

                    articles.append({
                        'title': title,
                        'content': snippet,
                        'url': link,
                        'source': source,
                        'date': article_date,
                        'ticker': ticker
                    })

                except Exception as e:
                    logger.debug(f"Error parsing Google News article: {e}")
                    continue

            logger.info(f"Found {len(articles)} articles from Google News")

        except Exception as e:
            logger.error(f"Error scraping Google News: {e}")

        return articles

    def _parse_date(self, date_str: str) -> datetime:
        """Parse Google News date format"""
        try:
            now = datetime.now()

            if 'ago' in date_str.lower():
                if 'hour' in date_str:
                    hours = int(re.search(r'(\d+)', date_str).group(1))
                    return now - timedelta(hours=hours)
                elif 'day' in date_str:
                    days = int(re.search(r'(\d+)', date_str).group(1))
                    return now - timedelta(days=days)
                elif 'minute' in date_str:
                    minutes = int(re.search(r'(\d+)', date_str).group(1))
                    return now - timedelta(minutes=minutes)
                elif 'week' in date_str:
                    weeks = int(re.search(r'(\d+)', date_str).group(1))
                    return now - timedelta(weeks=weeks)

        except Exception:
            pass

        return datetime.now()


class NewsAggregator:
    """Aggregates news from multiple sources"""

    def __init__(self, sources: Optional[List[str]] = None, rate_limit_delay: float = 1.5):
        """
        Args:
            sources: List of source names to use. Default: ['yahoo', 'google']
            rate_limit_delay: Delay between requests
        """
        self.rate_limit_delay = rate_limit_delay

        # Initialize scrapers
        self.scrapers = {
            'yahoo': YahooFinanceScraper(rate_limit_delay),
            'google': GoogleNewsScraper(rate_limit_delay),
        }

        # Use specified sources or all available
        self.active_sources = sources or list(self.scrapers.keys())

    def collect_news(self, ticker: str, company_name: str,
                     start_date: datetime, end_date: datetime,
                     output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Collect news from all active sources

        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            start_date: Start date for articles
            end_date: End date for articles
            output_path: Optional path to save CSV output

        Returns:
            DataFrame with all collected articles
        """
        all_articles = []

        for source_name in self.active_sources:
            scraper = self.scrapers.get(source_name)
            if not scraper:
                logger.warning(f"Unknown source: {source_name}")
                continue

            try:
                articles = scraper.scrape(ticker, company_name, start_date, end_date)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")

        # Create DataFrame
        df = pd.DataFrame(all_articles)

        if not df.empty:
            # Remove duplicates based on title similarity
            df = df.drop_duplicates(subset=['title'], keep='first')

            # Sort by date
            df = df.sort_values('date', ascending=False)

            logger.info(f"Total articles collected: {len(df)}")

            # Save to CSV if path provided
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved to {output_path}")
        else:
            logger.warning("No articles collected")

        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape news articles for stock analysis')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--company', required=True, help='Company name (e.g., "Apple Inc")')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--sources', nargs='+', default=['yahoo', 'google'],
                        help='News sources to use (default: yahoo google)')
    parser.add_argument('--delay', type=float, default=1.5,
                        help='Rate limit delay in seconds (default: 1.5)')

    args = parser.parse_args()

    # Parse dates
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Set default output path if not provided
    output_path = args.output or f"data/raw/{args.ticker}_news.csv"

    # Create aggregator and collect news
    aggregator = NewsAggregator(sources=args.sources, rate_limit_delay=args.delay)
    df = aggregator.collect_news(
        ticker=args.ticker,
        company_name=args.company,
        start_date=start,
        end_date=end,
        output_path=output_path
    )

    print(f"\nCollected {len(df)} articles")
    if not df.empty:
        print(f"\nSample articles:")
        print(df[['date', 'source', 'title']].head(10).to_string())
