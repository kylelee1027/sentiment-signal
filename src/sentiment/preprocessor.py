"""
Text preprocessing for sentiment analysis
"""

import re
import string
from typing import List, Optional


def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = True,
               remove_hashtags: bool = False, lowercase: bool = False) -> str:
    """
    Clean and normalize text for sentiment analysis

    Args:
        text: Raw text string
        remove_urls: Remove URLs from text
        remove_mentions: Remove @mentions
        remove_hashtags: Remove #hashtags
        lowercase: Convert to lowercase

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove mentions (@username)
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def remove_punctuation(text: str, keep_sentence_endings: bool = True) -> str:
    """
    Remove punctuation from text

    Args:
        text: Input text
        keep_sentence_endings: Keep periods, question marks, exclamation points

    Returns:
        Text without punctuation
    """
    if keep_sentence_endings:
        # Keep sentence-ending punctuation
        punctuation = string.punctuation.replace('.', '').replace('!', '').replace('?', '')
        text = text.translate(str.maketrans('', '', punctuation))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def remove_stopwords(text: str, custom_stopwords: Optional[List[str]] = None) -> str:
    """
    Remove common stopwords from text

    Args:
        text: Input text
        custom_stopwords: Optional list of custom stopwords

    Returns:
        Text without stopwords
    """
    # Basic English stopwords
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with'
    }

    if custom_stopwords:
        stopwords.update(custom_stopwords)

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]

    return ' '.join(filtered_words)


def normalize_financial_terms(text: str) -> str:
    """
    Normalize common financial terms and abbreviations

    Args:
        text: Input text

    Returns:
        Text with normalized financial terms
    """
    # Common financial term mappings
    replacements = {
        r'\bEPS\b': 'earnings per share',
        r'\bP/E\b': 'price to earnings',
        r'\bROI\b': 'return on investment',
        r'\bYoY\b': 'year over year',
        r'\bQoQ\b': 'quarter over quarter',
        r'\bIPO\b': 'initial public offering',
        r'\bM&A\b': 'mergers and acquisitions',
        r'\bCEO\b': 'chief executive officer',
        r'\bCFO\b': 'chief financial officer',
        r'\$(\d+)B\b': r'\1 billion dollars',
        r'\$(\d+)M\b': r'\1 million dollars',
        r'\$(\d+)K\b': r'\1 thousand dollars',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def prepare_for_sentiment(text: str, model_type: str = 'finbert') -> str:
    """
    Prepare text for sentiment analysis based on model type

    Args:
        text: Raw text
        model_type: Type of sentiment model ('finbert', 'vader', 'textblob')

    Returns:
        Preprocessed text
    """
    if model_type == 'finbert':
        # FinBERT works best with minimal preprocessing
        # Just clean URLs and normalize whitespace
        text = clean_text(text, remove_urls=True, remove_mentions=True,
                         remove_hashtags=False, lowercase=False)
        text = normalize_financial_terms(text)

    elif model_type == 'vader':
        # VADER is sensitive to capitalization and punctuation
        # Keep most formatting intact
        text = clean_text(text, remove_urls=True, remove_mentions=True,
                         remove_hashtags=False, lowercase=False)

    elif model_type == 'textblob':
        # TextBlob benefits from cleaner text
        text = clean_text(text, remove_urls=True, remove_mentions=True,
                         remove_hashtags=True, lowercase=True)
        text = remove_punctuation(text, keep_sentence_endings=True)

    return text


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate text to maximum length (for transformer models)

    Args:
        text: Input text
        max_length: Maximum number of characters

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + '...'


def extract_financial_entities(text: str) -> dict:
    """
    Extract financial entities from text (numbers, percentages, etc.)

    Args:
        text: Input text

    Returns:
        Dictionary of extracted entities
    """
    entities = {
        'percentages': [],
        'dollar_amounts': [],
        'numbers': []
    }

    # Extract percentages
    percentages = re.findall(r'\b(\d+\.?\d*)\s*%', text)
    entities['percentages'] = [float(p) for p in percentages]

    # Extract dollar amounts
    dollars = re.findall(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*([BMK])?', text)
    for amount, suffix in dollars:
        value = float(amount.replace(',', ''))
        if suffix == 'B':
            value *= 1_000_000_000
        elif suffix == 'M':
            value *= 1_000_000
        elif suffix == 'K':
            value *= 1_000
        entities['dollar_amounts'].append(value)

    # Extract general numbers
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    entities['numbers'] = [float(n) for n in numbers if n not in percentages]

    return entities
