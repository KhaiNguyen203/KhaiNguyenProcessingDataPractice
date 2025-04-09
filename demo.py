import pandas as pd
import matplotlib.pyplot as plt

BaTemp = 'Basel Temperature'
BaPrec = 'Basel Precipitation Total'

df = pd.read_csv('2014-2025_April_WeatherForecastDataInBasel.csv')
# chuyển dạng timestamp sang dạng yyyy-mm-dd
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M')
#Gộp phần giờ của mẫu số liệu chỉ lấy theo ngày
df['day'] = df['timestamp'].dt.to_period('D')
#tính trung bình dữ liệu của các giờ trong ngày
df_day = df.groupby('day').mean()
#dropna để loại bỏ các ô dữ liệu bị trống
print(df_day[['Basel Temperature', 'Basel Precipitation Total']].dropna())