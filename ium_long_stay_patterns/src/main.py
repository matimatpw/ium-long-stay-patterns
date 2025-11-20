import pandas as pd

chunksize = 100000  # 100k rows at a time
for chunk in pd.read_csv("data/raw/sessions.csv", chunksize=chunksize):
    print(chunk.head())
    # do analysis on this chunk