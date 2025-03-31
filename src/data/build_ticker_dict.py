#!/usr/bin/env python3
"""
build_ticker_dict.py

Script that reads a CSV with columns [Ticker, CompanyName],
generates a Python dictionary of the form:

{
  "AAPL": {"aapl", "apple inc", "apple"},
  "TSLA": {"tsla", "tesla inc", "tesla"},
  ...
}

Then saves that dictionary to a JSON file for easy loading later.

Usage:
------
  python build_ticker_dict.py --csv path/to/stock_symbols.csv --json path/to/ticker_dict.json

Where stock_symbols.csv might look like:
    Ticker,Company
    AAPL,Apple Inc.
    TSLA,Tesla Inc
    GOOG,Alphabet Inc

The script:
1) Reads the CSV.
2) For each ticker, produces a set of synonyms:
   - the lowercase ticker,
   - the company name in lowercase (punctuation removed),
   - the same name minus "Inc", "Corp", etc. (see common suffixes).
3) Writes the resulting dictionary to JSON.

You can then load the JSON in your notebook:
    import json
    with open("path/to/ticker_dict.json", "r") as f:
        data = json.load(f)
    ticker_dict = {k: set(v) for k, v in data.items()}
"""

import csv
import re
import json
import sys
import argparse

COMMON_SUFFIXES = [
    "inc", "inc.", "corp", "corp.", "co", "co.",
    "ltd", "ltd.", "plc", "plc.", "corporation",
    "group", "holdings", "holding", "company"
]

def remove_common_suffixes(name: str) -> str:
    """
    Removes common corporate suffixes from the END of a string if present.
    E.g. "Apple Inc" -> "Apple"
         "Acme Corp." -> "Acme"
    """
    tokens = name.strip().split()
    if not tokens:
        return name.strip()

    last_token = tokens[-1].lower()
    if last_token in COMMON_SUFFIXES:
        return " ".join(tokens[:-1])
    return name.strip()

def standardize_company_name(raw_name: str):
    """
    Given a raw company name, return a tuple of:
      (full_name, short_name)
    Both in lowercase, punctuation removed, etc.

    Example:
      "Apple Inc." -> ("apple inc", "apple")
    """
    lower_name = raw_name.lower()
    # remove punctuation but keep spaces
    full_name = re.sub(r"[^\w\s]", "", lower_name).strip()
    short_name = remove_common_suffixes(full_name)
    return (full_name, short_name)

def create_ticker_dictionary(csv_path: str):
    """
    Reads a CSV with columns [Ticker, CompanyName], returns a dict:
        {
          "AAPL": {"aapl", "apple inc", "apple"},
          ...
        }
    """
    ticker_dict = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip header (adjust if your CSV has no header)
        next(reader, None)

        for row in reader:
            if len(row) < 2:
                # skip if less than 2 columns
                continue
            ticker = row[0].strip()
            company_name = row[1].strip()

            # skip empty ticker
            if not ticker:
                continue

            synonyms = set()

            # 1) Lowercase version of ticker
            synonyms.add(ticker.lower())

            # 2) Full name, short name
            full_name, short_name = standardize_company_name(company_name)
            if full_name:
                synonyms.add(full_name)
            if short_name and short_name != full_name:
                synonyms.add(short_name)

            ticker_dict[ticker] = synonyms

    return ticker_dict

def save_ticker_dictionary_as_json(ticker_dict, output_path):
    """
    Writes the ticker dictionary to a JSON file.
    We must convert sets -> lists for JSON serialization.
    """
    data_for_json = {k: list(v) for k, v in ticker_dict.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_for_json, f, ensure_ascii=False, indent=2)
    print(f"Dictionary saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Build ticker-synonyms dictionary from CSV and save to JSON.")
    parser.add_argument("--csv", required=True, help="Path to input CSV (stock_symbols.csv)")
    parser.add_argument("--json", required=True, help="Output path for the JSON dictionary")
    args = parser.parse_args()

    csv_path = args.csv
    json_path = args.json

    # Build the dictionary
    ticker_dict = create_ticker_dictionary(csv_path)

    # Save as JSON
    save_ticker_dictionary_as_json(ticker_dict, json_path)

if __name__ == "__main__":
    main()
