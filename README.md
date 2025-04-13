# Financial News Sentiment Analysis and Stock Movement Prediction

## Project Overview

The project aims to analyze financial news sentiment and correlate it with stock price movements. It involves sentiment analysis (SA) on daily financial news using a fine-tuned DistilRoBERTa model and subsequent correlation with stock movements on the following day, enabling prediction of upward, downward, or neutral trends. The core domains include sentiment analysis, news processing, data cleaning and preparation, handling large datasets, NLP (Natural Language Processing), and custom model fine-tuning.

## Technology Stack
- **Programming Language:** Python
- **Data Handling:** PySpark, Pandas
- **Visualization:** Matplotlib
- **Machine Learning and NLP:** Hugging Face Transformers (DistilRoBERTa), SparkNLP, Torch
- **Web Scraping:** BeautifulSoup (bs4)
- **Financial Data:** yfinance
- **Cloud Storage:** boto3, botocore (AWS S3)
- **Data Formats:** CSV, Parquet

## Data Sources and Handling
- **Financial News Datasets:**
  - [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_news) (5 GB)
  - [Kaggle Financial News](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) (smaller dataset)
  - [Financial Phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank/tree/main) (for sentiment labeling)
- **Historical Stock Prices:** Retrieved with custom script using yfinance, stored as CSVs in `data/raw/price_data`
- **Processed Data:** Stored in Parquet format for optimized handling (`data/processed/`)
  - NLP processed news (`processed_news.parquet`, `processed_news_analyst.parquet`)
  - Sentiment analysis results (`sentiment_analyst.parquet`)
  - English-filtered news (`english_only.parquet`)
  - Ticker synonyms dictionary (`ticker_dict.json`)

## Configuration
- AWS S3 configurations and credentials managed via `.env` and basic YAML configurations (`config.yaml`).

## Project Structure
```
.
├── README.md
├── config.yaml
├── data
│   ├── raw
│   │   ├── price_data
│   │   ├── english_only.parquet
│   │   └── ticker_dict.json
│   └── processed
│       ├── processed_news.parquet
│       ├── processed_news_analyst.parquet
│       ├── sentiment_analyst.parquet
│       └── phraseBank
│           └── FinancialPhraseBank-v1.0
├── notebooks
│   ├── tests.ipynb
│   ├── test2.ipynb
│   └── testValle.ipynb
└── src
    ├── data
    │   ├── build_ticker_dict.py
    │   ├── data_preprocessing.py
    │   ├── data_scraper.py
    │   ├── historic_price_collector.py
    │   ├── load_financial_phrasebank.py
    │   ├── NLP_preprocessing.py
    │   └── s3_utils.py
    └── models
        ├── custom_model.py
        ├── fine_tuning.py
        ├── inference.py
        └── finetuned_distilroberta
```

## Jupyter Notebook Descriptions

### tests.ipynb
- Comprehensive data exploration, aggregation, and enhancement of the 5 GB dataset.
- Includes heuristic filtering of non-English articles and an unsuccessful attempt at data imputation for missing stock symbols (resulting in high false positives).
- Exploratory data analysis with visual plots of article distributions over time.
- Fine-tuning of DistilRoBERTa and saving of the trained model.
- Attempted sentiment prediction on the large dataset, terminated due to memory limitations caused by the requirement to convert Spark DataFrames to Pandas DataFrames, necessitating a switch to a smaller dataset.

### test2.ipynb
- Data exploration, preprocessing, and NLP application on a smaller dataset (~150 MB), allowing feasible processing and continuation of the project.
- Execution of sentiment analysis inference despite significant computational time.
- Implementation and comparative analysis of logistic regression and random forest classifiers.

### testValle.ipynb
- Focused on the collection of historical stock prices.
- Conducted quick tests and validations on the smaller NLP-processed dataset stored in Parquet format.

## Scripts and Functions
- **Data Handling Scripts:**
  - Build dictionary for stock tickers and synonyms
  - Collect and preprocess historical price data
  - Scrape stock symbols and full names

- **NLP and Model Fine-tuning:**
  - Fine-tune DistilRoBERTa using Financial Phrasebank
  - Perform sentiment analysis inference on financial news
  - Custom model layers: dropout, dense, nonlinear activations

## Model Performance
- **DistilRoBERTa Fine-tuned Model:**
  - Evaluation Accuracy: ~96.9%
  - Evaluation Loss: ~0.136
  - F1 Score: ~0.970

## Practical Applications
- Supporting trading decisions with sentiment insights
- Autonomous trading agent enhancement
- Predicting stock movements based on financial news sentiment

## Challenges
- Significant challenges in processing large datasets, requiring substantial computational resources.
- High runtime even when utilizing Spark, due to large data volume.

## Additional Resources
- Detailed analysis, preprocessing, and model tuning documented within provided Jupyter Notebooks (`tests.ipynb`, `test2.ipynb`, `testValle.ipynb`).

## Contributions
This repository is public, and contributions are welcome through pull requests or issue tracking. Performance enhancements, particularly regarding large data handling, are highly encouraged.

---

Developed and maintained by Andrés Felipe Fierro F.


