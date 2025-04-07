import requests
from bs4 import BeautifulSoup
import string
import time
import csv
import os

#/dhj
# All data is scraped from https://stock-screener.org/stock-list.aspx?alpha=A 
# /#


# Set the relative path (adjust as needed)
output_path = "../../data/raw/stock_symbols.csv"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Full path for debug
print(f"💾 Will save to: {os.path.abspath(output_path)}")

base_url = "https://stock-screener.org/stock-list.aspx?alpha="
http_headers = {
    "User-Agent": "Mozilla/5.0"
}

with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Symbol", "Company"])

    total_rows = 0

    for letter in string.ascii_uppercase:
        url = base_url + letter
        print(f"\n📄 Scraping letter: {letter}")

        try:
            response = requests.get(url, headers=http_headers)
            soup = BeautifulSoup(response.text, "html.parser")

            for outer_table in soup.find_all("table", class_="tablestyle"):
                styled = outer_table.find("table", class_="styled")
                if styled:
                    table_headers = [th.text.strip().lower() for th in styled.find("thead").find_all("th")]
                    if "symbol" in table_headers and "company" in table_headers:
                        for row in styled.find("tbody").find_all("tr"):
                            cols = row.find_all("td")
                            if len(cols) >= 3:
                                symbol = cols[0].text.strip()
                                company = cols[2].text.strip()
                                writer.writerow([symbol, company])
                                total_rows += 1
                                print(f"✔ {symbol}: {company}")
                        break  # found the correct table

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    # ✅ Force writing the buffer to disk
    csvfile.flush()
    os.fsync(csvfile.fileno())

print(f"\n✅ Done. {total_rows} rows saved to: {os.path.abspath(output_path)}")

# %%
