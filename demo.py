import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Spotify_2024_Global_Streaming_Data.csv')
print( df.groupby(['Artist', 'Monthly Listeners (Millions)'])['Total Streams (Millions)'].sum())