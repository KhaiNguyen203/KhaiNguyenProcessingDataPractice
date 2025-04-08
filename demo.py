import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Spotify_2024_Global_Streaming_Data.csv')
pd.set_option('display.max_columns', None)
print(df[df['Total Hours Streamed (Millions)'] == df['Total Hours Streamed (Millions)'].max()])

