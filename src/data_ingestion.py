import pandas as pd
import logging
from sklearn.model_selection import train_test_split

def load_and_process_data(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        # Cleaning steps here...
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train.to_csv(f"{output_path}/train.csv", index=False)
        test.to_csv(f"{output_path}/test.csv", index=False)
        logging.info("Data split and saved.")
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
