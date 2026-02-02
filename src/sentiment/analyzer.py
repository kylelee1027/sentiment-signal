"""
Sentiment Analysis Engine

Supports multiple sentiment analysis models:
- FinBERT: Pre-trained on financial communication
- VADER: Valence Aware Dictionary and sEntiment Reasoner
- TextBlob: Simple sentiment analysis library
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime

# Import preprocessing functions
from .preprocessor import prepare_for_sentiment, clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Base class for sentiment analyzers"""

    def __init__(self):
        self.model_name = "base"

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores
        """
        raise NotImplementedError

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Analyze sentiment for batch of texts

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of sentiment score dictionaries
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results

    def open_content(self, url: str) -> str:
        """
        Open and read content from a URL

        Args:
            url: URL to fetch content from

        Returns:
            Text content from the URL, or empty string on failure
        """
        import requests

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return ""

    def analyze_url(self, url: str) -> Dict[str, Union[float, str]]:
        """
        Fetch content from URL and analyze its sentiment

        Args:
            url: URL to fetch and analyze

        Returns:
            Dictionary with sentiment scores and the URL
        """
        logger.info(f"Fetching and analyzing content from {url}")

        # Fetch content
        content = self.open_content(url)

        if not content:
            logger.warning(f"No content retrieved from {url}")
            return {
                'url': url,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'error': 'Failed to fetch content'
            }

        # Extract text content from HTML
        from .preprocessor import clean_text
        text_content = clean_text(content)

        # Analyze sentiment
        sentiment_scores = self.analyze(text_content)
        sentiment_scores['url'] = url

        return sentiment_scores

    def analyze_urls_batch(self, urls: List[str], batch_size: int = 16) -> List[Dict[str, Union[float, str]]]:
        """
        Fetch content from multiple URLs and analyze their sentiment

        Args:
            urls: List of URLs to fetch and analyze
            batch_size: Number of URLs to process at once

        Returns:
            List of dictionaries with sentiment scores for each URL
        """
        logger.info(f"Fetching and analyzing content from {len(urls)} URLs")

        results = []
        for url in urls:
            result = self.analyze_url(url)
            results.append(result)

        return results


class FinBERTAnalyzer(SentimentAnalyzer):
    """FinBERT sentiment analyzer for financial text"""

    def __init__(self):
        super().__init__()
        self.model_name = "finbert"
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info("Loading FinBERT model...")

            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Set to evaluation mode
            self.model.eval()

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            logger.info(f"FinBERT model loaded successfully on {self.device}")

        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            logger.error("Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT

        Args:
            text: Input text

        Returns:
            Dictionary with 'positive', 'negative', 'neutral' scores and 'compound'
        """
        import torch

        if not text or not text.strip():
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

        # Preprocess text
        text = prepare_for_sentiment(text, model_type='finbert')

        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [positive, negative, neutral]
        scores = predictions[0].cpu().numpy()

        positive_score = float(scores[0])
        negative_score = float(scores[1])
        neutral_score = float(scores[2])

        # Calculate compound score (-1 to 1)
        compound = positive_score - negative_score

        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score,
            'compound': compound
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Analyze sentiment for batch of texts (more efficient for FinBERT)

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of sentiment dictionaries
        """
        import torch

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Preprocess batch
            cleaned_batch = [prepare_for_sentiment(text, model_type='finbert') for text in batch]

            # Tokenize batch
            inputs = self.tokenizer(cleaned_batch, return_tensors='pt',
                                   truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process results
            for scores in predictions.cpu().numpy():
                positive_score = float(scores[0])
                negative_score = float(scores[1])
                neutral_score = float(scores[2])
                compound = positive_score - negative_score

                results.append({
                    'positive': positive_score,
                    'negative': negative_score,
                    'neutral': neutral_score,
                    'compound': compound
                })

        return results


class VADERAnalyzer(SentimentAnalyzer):
    """VADER sentiment analyzer"""

    def __init__(self):
        super().__init__()
        self.model_name = "vader"
        self._load_model()

    def _load_model(self):
        """Load VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER analyzer loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import VADER: {e}")
            logger.error("Install with: pip install vaderSentiment")
            raise

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores
        """
        if not text or not text.strip():
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

        # Preprocess text
        text = prepare_for_sentiment(text, model_type='vader')

        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)

        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }


class TextBlobAnalyzer(SentimentAnalyzer):
    """TextBlob sentiment analyzer"""

    def __init__(self):
        super().__init__()
        self.model_name = "textblob"

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores
        """
        try:
            from textblob import TextBlob
        except ImportError as e:
            logger.error(f"Failed to import TextBlob: {e}")
            logger.error("Install with: pip install textblob")
            raise

        if not text or not text.strip():
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

        # Preprocess text
        text = prepare_for_sentiment(text, model_type='textblob')

        # Get TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1

        # Convert to format similar to other analyzers
        if polarity > 0:
            positive = polarity
            negative = 0.0
            neutral = 1.0 - polarity
        elif polarity < 0:
            positive = 0.0
            negative = abs(polarity)
            neutral = 1.0 - abs(polarity)
        else:
            positive = 0.0
            negative = 0.0
            neutral = 1.0

        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'compound': polarity
        }


class SentimentPipeline:
    """Sentiment analysis pipeline for processing articles"""

    def __init__(self, model_type: str = 'finbert'):
        """
        Initialize sentiment pipeline

        Args:
            model_type: Type of model to use ('finbert', 'vader', 'textblob')
        """
        self.model_type = model_type.lower()

        # Initialize appropriate analyzer
        if self.model_type == 'finbert':
            self.analyzer = FinBERTAnalyzer()
        elif self.model_type == 'vader':
            self.analyzer = VADERAnalyzer()
        elif self.model_type == 'textblob':
            self.analyzer = TextBlobAnalyzer()
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: finbert, vader, textblob")

        logger.info(f"Sentiment pipeline initialized with {self.model_type}")

    def analyze_articles(self, df: pd.DataFrame, text_column: str = 'content',
                        title_column: str = 'title', batch_size: int = 16) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of articles

        Args:
            df: DataFrame with articles
            text_column: Column name containing article text
            title_column: Column name containing article titles
            batch_size: Batch size for processing

        Returns:
            DataFrame with added sentiment columns
        """
        logger.info(f"Analyzing sentiment for {len(df)} articles using {self.model_type}")

        # Combine title and content for analysis
        texts = []
        for idx, row in df.iterrows():
            title = str(row.get(title_column, ''))
            content = str(row.get(text_column, ''))
            combined_text = f"{title}. {content}" if title else content
            texts.append(combined_text)

        # Analyze sentiment
        results = self.analyzer.analyze_batch(texts, batch_size=batch_size)

        # Add results to DataFrame
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]

        # Add sentiment label based on compound score
        df['sentiment_label'] = df['sentiment_compound'].apply(self._get_label)

        logger.info(f"Sentiment analysis complete")

        return df

    def _get_label(self, compound_score: float) -> str:
        """
        Get sentiment label from compound score

        Args:
            compound_score: Compound sentiment score

        Returns:
            Sentiment label
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze sentiment of news articles')
    parser.add_argument('--input', required=True, help='Input CSV file with articles')
    parser.add_argument('--output', required=True, help='Output CSV file with sentiment scores')
    parser.add_argument('--model', default='finbert', choices=['finbert', 'vader', 'textblob'],
                       help='Sentiment model to use (default: finbert)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing (default: 16)')
    parser.add_argument('--text-column', default='content',
                       help='Column name containing article text')
    parser.add_argument('--title-column', default='title',
                       help='Column name containing article titles')

    args = parser.parse_args()

    # Load articles
    logger.info(f"Loading articles from {args.input}")
    df = pd.read_csv(args.input)

    # Initialize pipeline
    pipeline = SentimentPipeline(model_type=args.model)

    # Analyze sentiment
    df = pipeline.analyze_articles(df, text_column=args.text_column,
                                   title_column=args.title_column,
                                   batch_size=args.batch_size)

    # Save results
    df.to_csv(args.output, index=False)
    logger.info(f"Sentiment analysis complete. Results saved to {args.output}")

    # Print summary
    print("\nSentiment Distribution:")
    print(df['sentiment_label'].value_counts())
    print("\nAverage Sentiment Scores:")
    print(df[['sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_compound']].mean())
