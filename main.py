#!/usr/bin/env python3
"""
Main orchestrator for the Financial Sentiment Project.

Usage:
    python main.py --collect-prices
    python main.py --process-data
    python main.py --collect-prices --process-data
"""

import argparse
import yaml
import os

# Import your refactored scripts
from src.data.historic_price_collector import run_collection_flow
from src.data.data_preprocessing import run_preprocessing_flow

def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Financial Sentiment Pipeline Orchestrator")
    parser.add_argument("--collect-prices", action="store_true", help="Collect historical prices.")
    parser.add_argument("--process-data", action="store_true", help="Clean and preprocess data.")
    # In the future, you might add: --train-model, --run-inference, etc.

    args = parser.parse_args()

    # Load config
    config = load_config("config.yaml")

    # Decide which steps to run
    if args.collect_prices:
        run_collection_flow(config)

    if args.process_data:
        run_preprocessing_flow(config)

    # If no flags are given, run all steps in sequence
    if not any(vars(args).values()):
        run_collection_flow(config)
        run_preprocessing_flow(config)

if __name__ == "__main__":
    main()
