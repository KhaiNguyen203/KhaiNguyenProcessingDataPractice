import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Spotify_2024_Global_Streaming_Data.csv')
print(df.sort_values("Total Streams (Millions)").head(10), df[['Artist', 'Total Streams (Millions)']])