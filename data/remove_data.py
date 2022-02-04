from os import replace
import pandas as pd

input_file = "./data/yellow_tripdata_2021-01.csv"
df = pd.read_csv(input_file, header=0, low_memory=False)
new_df = df.sample(frac=0.5, replace=False, random_state=97)
new_df.to_csv('yellow_tripdata_2021-012.csv')