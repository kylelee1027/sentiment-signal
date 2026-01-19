# Stock Sentiment Predictor

A machine learning system that combines web scraping, sentiment analysis, and recurrent neural networks to predict stock price movements based on public sentiment and historical price patterns.

## 🎯 Project Overview

This project aims to assist with investment decisions by:
1. Scraping public sentiment data about companies from various online sources
2. Analyzing sentiment to gauge public perception over specific time periods
3. Training a neural network that combines sentiment data with historical stock prices
4. Predicting future stock price movements
5. Backtesting the strategy to validate performance

## 🏗️ System Architecture

```
Data Collection → Sentiment Analysis → Feature Engineering → ML Model → Backtesting
      ↓                  ↓                    ↓                ↓            ↓
  Web Scraper      NLP Processing      Technical Indicators   RNN+Attention  Performance
  (News, Social)   (FinBERT/VADER)    (RSI, MACD, etc.)      Prediction     Metrics
```

## 🔧 Core Components

### 1. Web Scraper with Sentiment Analysis
- **Purpose**: Gather public sentiment data about target companies
- **Features**:
  - Search by company name and date range
  - Aggregate data from multiple sources (news sites, social media, financial forums)
  - Perform sentiment analysis on collected text
  - Output: Sentiment score (positive/negative/neutral) for specified time periods

### 2. RNN with Self-Attention Mechanism
- **Purpose**: Predict stock price movements using historical data and sentiment
- **Architecture**:
  - Recurrent Neural Network (LSTM/GRU) with attention layers
  - Multi-modal input: price patterns + sentiment scores
  - Output: Predicted percentage change over X time period
- **Training Data**:
  - Historical stock prices (OHLCV data)
  - Technical indicators
  - Daily aggregated sentiment scores
  - Market context (indices, sector performance)
- **Infrastructure**: Cloud deployment on GCP Vertex AI for compute-intensive training

### 3. Backtesting Module
- **Purpose**: Validate strategy performance on historical data
- **Features**:
  - Simulate trades based on model predictions
  - Calculate key performance metrics
  - Compare against baseline strategies
  - Identify potential overfitting or data leakage

## 📊 Key Metrics

The system evaluates performance using:
- **Total Return**: Overall profit/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Sortino Ratio**: Downside risk-adjusted returns

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow/PyTorch
Beautiful Soup / Scrapy
Transformers (for FinBERT)
pandas, numpy, scikit-learn
backtesting library (backtrader/vectorbt)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/stock-sentiment-predictor.git
cd stock-sentiment-predictor

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Quick Start
```bash
# 1. Collect data for a company
python src/scraping/collect_data.py --ticker AAPL --start-date 2023-01-01 --end-date 2024-01-01

# 2. Run sentiment analysis
python src/sentiment/analyzer.py --input data/raw/AAPL_news.csv --output data/processed/AAPL_sentiment.csv

# 3. Train the model
python src/models/train.py --ticker AAPL --epochs 100 --batch-size 32

# 4. Run backtesting
python src/backtesting/backtest.py --ticker AAPL --strategy model_predictions --start-date 2024-01-01
```

## 📁 Project Structure

```
stock-sentiment-predictor/
├── data/
│   ├── raw/                    # Scraped raw data
│   ├── processed/              # Cleaned and featured data
│   └── models/                 # Saved model weights
├── src/
│   ├── scraping/
│   │   ├── news_scraper.py     # News article scraper
│   │   ├── social_scraper.py   # Social media scraper
│   │   └── utils.py            # Scraping utilities
│   ├── sentiment/
│   │   ├── analyzer.py         # Sentiment analysis engine
│   │   └── preprocessor.py     # Text preprocessing
│   ├── features/
│   │   ├── engineering.py      # Feature creation
│   │   └── technical.py        # Technical indicators
│   ├── models/
│   │   ├── rnn_attention.py    # Model architecture
│   │   ├── train.py            # Training script
│   │   └── predict.py          # Inference script
│   └── backtesting/
│       ├── backtest.py         # Backtesting engine
│       └── metrics.py          # Performance metrics
├── notebooks/                  # Jupyter notebooks for EDA
├── tests/                      # Unit tests
├── config/
│   └── config.yaml            # Configuration file
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## 🔍 Data Sources

The system can scrape sentiment data from:
- **News**: Yahoo Finance, Google News, Reuters, Bloomberg
- **Social Media**: Reddit (r/wallstreetbets, r/stocks), Twitter/X
- **Financial Forums**: StockTwits, Seeking Alpha
- **SEC Filings**: EDGAR for fundamental analysis

## 🧠 Model Details

### Input Features
- **Price Data**: Open, High, Low, Close, Volume (OHLCV)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Sentiment Scores**: Daily aggregated sentiment from multiple sources
- **Market Context**: S&P 500, sector indices, VIX

### Architecture
```
Input Layer (60-day lookback window)
    ↓
LSTM Layer (128 units) + Dropout
    ↓
Self-Attention Layer
    ↓
LSTM Layer (64 units) + Dropout
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (1 unit, predicted % change)
```

### Training Strategy
- **Lookback Window**: 60 days of historical data
- **Prediction Horizon**: 5 days ahead (configurable)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout, L2 regularization

## ⚠️ Important Warnings

### Financial Risks
- **Past performance does not guarantee future results**
- Markets are inherently unpredictable and influenced by countless factors
- This tool is for educational and research purposes
- **Always start with paper trading** before risking real capital
- Consider transaction costs, slippage, and market impact

### Technical Challenges
- **Data Quality**: Scraped data may be noisy or biased
- **Model Drift**: Markets evolve; models require regular retraining
- **Overfitting**: High risk when training on financial data
- **Look-Ahead Bias**: Must be extremely careful about data leakage
- **Survivorship Bias**: Only testing on existing companies skews results

### Legal & Compliance
- Ensure compliance with data scraping policies and Terms of Service
- Be aware of securities trading regulations in your jurisdiction
- Respect rate limits on APIs
- Consider insider trading laws when using certain data sources

## 🛣️ Roadmap

### Phase 1: Data Pipeline (Weeks 1-3)
- [ ] Implement web scrapers for news and social media
- [ ] Build sentiment analysis pipeline
- [ ] Collect historical data for initial 10-20 stocks
- [ ] Set up data storage and versioning

### Phase 2: Feature Engineering (Weeks 4-5)
- [ ] Calculate technical indicators
- [ ] Create sentiment aggregation logic
- [ ] Implement train/validation/test splits
- [ ] Feature normalization and scaling

### Phase 3: Model Development (Weeks 6-9)
- [ ] Baseline models (buy-and-hold, moving averages)
- [ ] LSTM with attention implementation
- [ ] Hyperparameter tuning
- [ ] Cross-validation on multiple stocks

### Phase 4: Backtesting (Weeks 10-12)
- [ ] Implement backtesting framework
- [ ] Run historical simulations
- [ ] Performance analysis and metrics
- [ ] Comparison with benchmarks

### Phase 5: Deployment (Weeks 13-15)
- [ ] Cloud infrastructure setup (GCP Vertex AI)
- [ ] Inference pipeline
- [ ] Monitoring and alerting
- [ ] Paper trading validation

## 📚 Resources & References

### Machine Learning for Finance
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) by Marcos López de Prado
- [Machine Learning for Algorithmic Trading](https://www.packtpub.com/product/machine-learning-for-algorithmic-trading-second-edition/9781839217715) by Stefan Jansen

### Sentiment Analysis
- [FinBERT: Pre-trained on Financial Communication](https://github.com/ProsusAI/finBERT)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

### Backtesting Frameworks
- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [VectorBT](https://vectorbt.dev/)

---

**⭐ If you find this project useful, please consider giving it a star!**
