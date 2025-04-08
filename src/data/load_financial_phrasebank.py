import pandas as pd
import chardet

def load_financial_phrasebank(file_path, delimiter="@"):
    """
    Load the FinancialPhraseBank data from a file.
    Assumes each line is in the format: sentence@sentiment

    Args:
        file_path (str): Path to the text file.
        delimiter (str): Delimiter separating sentence and sentiment (default "@").

    Returns:
        pd.DataFrame: A DataFrame with columns "text" and "label".
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(100000)  # read a sample
        result = chardet.detect(raw_data)
        print("Detected encoding:", result["encoding"])

    data = []
    with open(file_path, "r", encoding=result["encoding"]) as f:
        for line in f:
            # Remove trailing newline and split on the delimiter
            parts = line.strip().split(delimiter)
            if len(parts) == 2:
                sentence, sentiment = parts
                data.append({"text": sentence.strip(), "label": sentiment.strip().lower()})
    df = pd.DataFrame(data)
    return df

def map_labels(df):
    """
    Map string sentiment labels to numerical values.
    For example: negative->0, neutral->1, positive->2.

    Args:
        df (pd.DataFrame): DataFrame with a 'label' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'label_id' column.
    """
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label_id"] = df["label"].map(mapping)
    return df

# Example usage:
if __name__ == "__main__":
    file_path = "../../data/processed/phraseBank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"  # Adjust path as needed
    df_fp = load_financial_phrasebank(file_path)
    df_fp = map_labels(df_fp)
    print(df_fp.head())
