#!/usr/bin/env python3
"""
Historical Price Data Collector using yfinance

This script downloads historical price data for a list of tickers between
specified start and end dates, then saves the data as CSV files in a designated
output folder. It includes error handling and logging for robustness.

Requirements:
    - yfinance (pip install yfinance)
    - pandas (pip install pandas)
"""

import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def download_price_data(ticker, start_date, end_date, interval="1d"):
    """
    Downloads historical price data for a given ticker.

    Args:
        ticker (str): The stock symbol (e.g., 'AAPL').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
        pd.DataFrame: DataFrame with the historical data or None if error.
    """
    try:
        logging.info(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            logging.warning(f"No data found for {ticker}.")
            return None
        data.reset_index(inplace=True)
        logging.info(f"Successfully downloaded data for {ticker}.")
        return data
    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {e}")
        return None

def save_data_to_csv(df, output_path, ticker):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Directory where CSV file will be stored.
        ticker (str): The stock symbol (used in file naming).
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, f"{ticker}_historical.csv")
        df.to_csv(file_path, index=False)
        logging.info(f"Data for {ticker} saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving data for {ticker}: {e}")

def collect_historical_data(tickers, start_date, end_date, output_path, interval="1d"):
    """
    Collects and saves historical data for a list of tickers.

    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        output_path (str): Folder where CSV files will be saved.
        interval (str): Data interval (default is '1d').
    """
    for ticker in tickers:
        data = download_price_data(ticker, start_date, end_date, interval)
        if data is not None:
            save_data_to_csv(data, output_path, ticker)
        else:
            logging.warning(f"Skipping saving for {ticker} due to download error or no data.")


def run_collection_flow(config):
    """
    Main entry point for historical price collection using a config dictionary.
    """
    tickers = config.get("tickers", [])
    start_date = config.get("start_date", "2014-01-01")
    end_date = config.get("end_date", datetime.today().strftime('%Y-%m-%d'))
    interval = config.get("interval", "1d")

    # Where to store raw data locally
    raw_data_path = config["local_paths"]["raw_data"]
    price_data_subfolder = os.path.join(raw_data_path, "price_data")

    logging.info("Starting historical price data collection...")
    logging.info(f"Tickers: {tickers}")
    logging.info(f"Date range: {start_date} to {end_date}")
    logging.info(f"Interval: {interval}")
    logging.info(f"Output path: {price_data_subfolder}")

    collect_historical_data(tickers, start_date, end_date, price_data_subfolder, interval)

    logging.info("Historical price data collection completed.")

if __name__ == "__main__":
    # Optional: If you run this script standalone, define a default config here
    default_config = {
        "tickers": ["AAPL", "GOOGL", "MSFT"],
        "start_date": "2014-01-01",
        "end_date": datetime.today().strftime('%Y-%m-%d'),
        "interval": "1d",
        "local_paths": {
            "raw_data": "../../data/raw"
        }
    }
    run_collection_flow(default_config)