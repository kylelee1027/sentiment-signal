"""
Utility functions for web scraping
"""

import re
import time
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content

    Args:
        text: Raw text string

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_domain(url: str) -> str:
    """
    Extract domain from URL

    Args:
        url: Full URL string

    Returns:
        Domain name
    """
    match = re.search(r'https?://([^/]+)', url)
    return match.group(1) if match else ""


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol to uppercase without special characters

    Args:
        ticker: Stock ticker symbol

    Returns:
        Normalized ticker
    """
    return ticker.upper().replace('.', '-').strip()


def generate_article_id(title: str, url: str, date: datetime) -> str:
    """
    Generate unique ID for article based on title, URL, and date

    Args:
        title: Article title
        url: Article URL
        date: Publication date

    Returns:
        Unique hash ID
    """
    content = f"{title}|{url}|{date.isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()


def is_relevant_article(title: str, content: str, keywords: list) -> bool:
    """
    Check if article is relevant based on keywords

    Args:
        title: Article title
        content: Article content
        keywords: List of keywords to match

    Returns:
        True if article contains any keyword
    """
    text = (title + " " + content).lower()

    for keyword in keywords:
        if keyword.lower() in text:
            return True

    return False


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on failure with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for exponential delay

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise

                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper
    return decorator


def rate_limit(calls_per_second: float):
    """
    Decorator to enforce rate limiting

    Args:
        calls_per_second: Maximum number of calls per second

    Returns:
        Decorated function
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed

            if left_to_wait > 0:
                time.sleep(left_to_wait)

            result = func(*args, **kwargs)
            last_called[0] = time.time()

            return result

        return wrapper
    return decorator


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate that date range is logical

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        True if valid
    """
    if start_date > end_date:
        logger.error("Start date must be before end date")
        return False

    if end_date > datetime.now():
        logger.warning("End date is in the future")

    return True


def parse_relative_date(date_str: str) -> Optional[datetime]:
    """
    Parse relative date strings like '2 hours ago', '3 days ago'

    Args:
        date_str: Relative date string

    Returns:
        Datetime object or None if parsing fails
    """
    from datetime import timedelta

    now = datetime.now()
    date_str = date_str.lower().strip()

    # Pattern: "X time_unit ago"
    pattern = r'(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago'
    match = re.search(pattern, date_str)

    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == 'second':
        return now - timedelta(seconds=amount)
    elif unit == 'minute':
        return now - timedelta(minutes=amount)
    elif unit == 'hour':
        return now - timedelta(hours=amount)
    elif unit == 'day':
        return now - timedelta(days=amount)
    elif unit == 'week':
        return now - timedelta(weeks=amount)
    elif unit == 'month':
        return now - timedelta(days=amount * 30)  # Approximate
    elif unit == 'year':
        return now - timedelta(days=amount * 365)  # Approximate

    return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters for filenames
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:max_length - len(ext) - 1] + ('.' + ext if ext else '')

    return filename


def get_stock_keywords(ticker: str, company_name: str) -> list:
    """
    Generate list of keywords for filtering relevant articles

    Args:
        ticker: Stock ticker symbol
        company_name: Company name

    Returns:
        List of keywords
    """
    keywords = [
        ticker.upper(),
        company_name,
        'stock',
        'shares',
        'earnings',
        'revenue',
        'profit',
        'loss',
        'market',
        'investor',
        'trading'
    ]

    # Add company name variations
    if company_name:
        # Remove common suffixes
        base_name = re.sub(r'\s+(Inc|Corp|Corporation|Ltd|Limited|LLC|Co|Company)\.?$',
                          '', company_name, flags=re.IGNORECASE)
        keywords.append(base_name)

    return list(set(keywords))  # Remove duplicates


def chunk_date_range(start_date: datetime, end_date: datetime, chunk_days: int = 30):
    """
    Split date range into smaller chunks for efficient scraping

    Args:
        start_date: Start date
        end_date: End date
        chunk_days: Number of days per chunk

    Yields:
        Tuples of (chunk_start, chunk_end)
    """
    from datetime import timedelta

    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=chunk_days), end_date)
        yield (current, chunk_end)
        current = chunk_end + timedelta(days=1)


def format_article_summary(article: Dict[str, Any]) -> str:
    """
    Format article data for display

    Args:
        article: Dictionary containing article data

    Returns:
        Formatted string
    """
    return f"""
    Title: {article.get('title', 'N/A')}
    Source: {article.get('source', 'N/A')}
    Date: {article.get('date', 'N/A')}
    URL: {article.get('url', 'N/A')}
    Content: {article.get('content', 'N/A')[:200]}...
    """.strip()


def detect_language(text: str) -> str:
    """
    Simple language detection (English vs non-English)

    Args:
        text: Text to analyze

    Returns:
        'en' for English, 'other' for non-English
    """
    # Simple heuristic: check for common English words
    english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with']

    text_lower = text.lower()
    english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')

    return 'en' if english_count >= 3 else 'other'


class ScrapingStats:
    """Track scraping statistics"""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.articles_collected = 0
        self.start_time = time.time()

    def record_request(self, success: bool = True):
        """Record a request attempt"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def record_articles(self, count: int):
        """Record number of articles collected"""
        self.articles_collected += count

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        elapsed = time.time() - self.start_time

        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            'articles_collected': self.articles_collected,
            'elapsed_time': elapsed,
            'requests_per_minute': (self.total_requests / elapsed * 60) if elapsed > 0 else 0
        }

    def print_summary(self):
        """Print statistics summary"""
        stats = self.get_summary()

        print("\n" + "="*50)
        print("SCRAPING STATISTICS")
        print("="*50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Articles Collected: {stats['articles_collected']}")
        print(f"Elapsed Time: {stats['elapsed_time']:.1f}s")
        print(f"Requests/Minute: {stats['requests_per_minute']:.1f}")
        print("="*50 + "\n")
