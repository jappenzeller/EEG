import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def write_submission(df, filename='submission.csv'):
    df.to_csv(filename, index=False)
    print(f"Wrote {filename}")

def kaggle_submit(filename, message, competition):
    api = KaggleApi(); api.authenticate()
    api.competition_submit(filename, message, competition)
    print("Submitted to Kaggle")
