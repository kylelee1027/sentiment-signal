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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
            # Build URL with date range filter
            # Format: tbs=cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY
            date_filter = f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"
            url = f"{self.base_url}?q={quote(query)}&tbm=nws&tbs={date_filter}"
            logger.info(f"Fetching URL: {url}")
            response = self._make_request(url)

            if not response:
                return articles

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article containers - actual structure: div class="Gx5Zad xpd EtOod pkphOe"
            article_containers = soup.find_all('div', class_='Gx5Zad')

            logger.info(f"Found {len(article_containers)} article containers")

            for idx, container in enumerate(article_containers[:30], 1):
                try:
                    # Find the main link (<a> tag that wraps the article)
                    link_elem = container.find('a', href=True)
                    if not link_elem:
                        logger.debug(f"Article {idx}: No link found")
                        continue

                    link = link_elem['href']

                    # Skip navigation links (Next, etc)
                    if 'nBDE1b' in link_elem.get('class', []):
                        continue

                    # Extract title - it's in div with class "ilUpNd UFvD1"
                    title_elem = container.find('div', class_='ilUpNd UFvD1')
                    if not title_elem:
                        # Try alternative - h3 tag
                        title_elem = container.find('h3')

                    if not title_elem:
                        logger.debug(f"Article {idx}: No title found")
                        continue

                    title = title_elem.get_text(strip=True)

                    # Extract source - div with class "ilUpNd BamJPe aSRlid XR4uSe"
                    source_elem = container.find('div', class_='BamJPe')
                    source = source_elem.get_text(strip=True) if source_elem else "Google News"

                    # Extract snippet/content - div with class "ilUpNd H66NU aSRlid"
                    snippet_elem = container.find('div', class_='H66NU')
                    snippet = ""
                    if snippet_elem:
                        snippet_text = snippet_elem.get_text(strip=True)
                        # Remove the date from the snippet if it's there
                        # Date is usually at the end after the content
                        snippet = snippet_text

                    # Extract date - span with class "UK5aid MDvRSc"
                    date_elem = container.find('span', class_='UK5aid')
                    article_date = datetime.now()

                    if date_elem:
                        date_str = date_elem.get_text(strip=True)
                        article_date = self._parse_date(date_str)
                        # Remove date from snippet if it was included
                        if snippet and date_str in snippet:
                            snippet = snippet.replace(date_str, '').strip()

                    # Add article (Google already filtered by date range)
                    articles.append({
                        'title': title,
                        'content': snippet,
                        'url': link,
                        'source': source,
                        'date': article_date,
                        'ticker': ticker
                    })
                    logger.debug(f"Article {idx}: Successfully parsed - {title[:50]}...")

                except Exception as e:
                    logger.debug(f"Error parsing article {idx}: {e}")
                    continue

            logger.info(f"Collected {len(articles)} articles from Google News (within date range)")

        except Exception as e:
            logger.error(f"Error scraping Google News: {e}")

        return articles

    def _parse_date(self, date_str: str) -> datetime:
        """Parse Google News date format (relative dates like '14 hours ago')"""
        try:
            now = datetime.now()
            date_lower = date_str.lower().strip()

            if 'ago' in date_lower:
                # Extract number from string
                num_match = re.search(r'(\d+)', date_str)
                if not num_match:
                    return now

                num = int(num_match.group(1))

                if 'hour' in date_lower:
                    return now - timedelta(hours=num)
                elif 'day' in date_lower:
                    return now - timedelta(days=num)
                elif 'minute' in date_lower:
                    return now - timedelta(minutes=num)
                elif 'week' in date_lower:
                    return now - timedelta(weeks=num)
                elif 'month' in date_lower:
                    return now - timedelta(days=num * 30)  # Approximate
                elif 'year' in date_lower:
                    return now - timedelta(days=num * 365)  # Approximate

            # Try to parse absolute dates if present
            for fmt in ['%B %d, %Y', '%b %d, %Y', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

        except Exception as e:
            logger.debug(f"Error parsing date '{date_str}': {e}")

        return datetime.now()


class NewsAPIScraper(BaseNewsScraper):
    """
    Scraper for NewsAPI.org
    
    This will also only be used for gathering recent news articles. API key required. Once upgraded to paid plan, 
    can use for historical data as well.
    """

    def __init__(self, rate_limit_delay: float = 1.0):
        super().__init__(rate_limit_delay)
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"

        if not self.api_key:
            logger.warning("NEWS_API_KEY not found in environment variables")

    def scrape(self, ticker: str, company_name: str,
               start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Scrape news from NewsAPI

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for search query
            start_date: Start date for news articles
            end_date: End date for news articles

        Returns:
            List of dictionaries containing article data
        """
        articles = []

        if not self.api_key:
            logger.error("Cannot scrape NewsAPI without API key")
            return articles

        logger.info(f"Fetching NewsAPI news for {company_name}")

        try:
            time.sleep(self.rate_limit_delay)

            # Build query - search for company name and stock-related terms
            query = f'"{company_name}" OR {ticker} AND (stock OR shares OR trading OR market)'

            # API parameters
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,  # Max results per request
                'apiKey': self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return articles

            total_results = data.get('totalResults', 0)
            logger.info(f"NewsAPI returned {total_results} total results")

            for item in data.get('articles', []):
                try:
                    # Parse publication date
                    published_at = item.get('publishedAt', '')
                    if published_at:
                        # NewsAPI returns ISO 8601 format: 2024-01-19T10:30:00Z
                        article_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        # Convert to naive datetime for consistency
                        article_date = article_date.replace(tzinfo=None)
                    else:
                        article_date = datetime.now()

                    title = item.get('title', '')
                    description = item.get('description', '')
                    content = item.get('content', '')
                    url = item.get('url', '')
                    source_name = item.get('source', {}).get('name', 'NewsAPI')
                    author = item.get('author', '')

                    # Skip articles with removed content
                    if title == '[Removed]' or not title:
                        continue

                    # Combine description and content for fuller text
                    full_content = description
                    if content and content != description:
                        full_content = f"{description} {content}" if description else content

                    articles.append({
                        'title': title,
                        'content': full_content,
                        'url': url,
                        'source': source_name,
                        'date': article_date,
                        'ticker': ticker,
                        'author': author
                    })

                except Exception as e:
                    logger.debug(f"Error parsing NewsAPI article: {e}")
                    continue

            logger.info(f"Collected {len(articles)} articles from NewsAPI")

        except requests.RequestException as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with NewsAPI: {e}")

        return articles


class NewsAggregator:
    """Aggregates news from multiple sources"""

    def __init__(self, sources: Optional[List[str]] = None, rate_limit_delay: float = 1.5):
        """
        Args:
            sources: List of source names to use. Default: ['yahoo', 'google', 'newsapi']
            rate_limit_delay: Delay between requests
        """
        self.rate_limit_delay = rate_limit_delay

        # Initialize scrapers
        self.scrapers = {
            'yahoo': YahooFinanceScraper(rate_limit_delay),
            'google': GoogleNewsScraper(rate_limit_delay),
            'newsapi': NewsAPIScraper(rate_limit_delay),
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
                if source_name == 'yahoo':
                    articles = scraper.scrape(ticker)
                else:
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
    parser.add_argument('--sources', nargs='+', default=['yahoo', 'google', 'newsapi'],
                        help='News sources to use (default: yahoo google newsapi)')
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