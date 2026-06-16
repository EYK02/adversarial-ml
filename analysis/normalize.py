# analysis/normalize.py

import pandas as pd
from src.logging.schema import normalize_row

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    rows = [normalize_row(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows)