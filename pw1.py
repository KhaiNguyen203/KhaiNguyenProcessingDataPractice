import pandas as pd
import matplotlib.pyplot as plt
# Đây là module để điều chỉnh các mốc chia (ticks) trên trục (x hoặc y).
import matplotlib.ticker as ticker
# Đây là module chuyên để xử lý các trục thời gian (ngày/tháng/năm).
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split

BaTemp = 'Basel Temperature'
BaPrec = 'Basel Precipitation Total'

df = pd.read_csv('2014-2025_WeatherForecastDataInBasel.csv')

# chuyển dạng timestamp sang dạng yyyy-mm-dd
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M')
#Gộp phần giờ của mẫu số liệu chỉ lấy theo ngày
df['day'] = pd.to_datetime(df['timestamp'].dt.date) # hoặc .dt.to_period('D') cũng được

df = df.sort_values(by="day")
# set index là ngày ở đây để có thể slice mẫu dữ liệu theo ngày
df = df.set_index('day')
# chia dữ liệu thành 2 phần để train và để test
train_df = df[:'2024-12-31']
test_df = df['2025-01-01':]
#tính trung bình dữ liệu của các giờ trong ngày
df_day = train_df.groupby('day')[['Basel Temperature', 'Basel Precipitation Total']].mean()
#Xóa các ô dữ liệu trống
df_day = df_day[['Basel Temperature', 'Basel Precipitation Total']].dropna()
# Chuyển 'day' từ index thành cột
df_day = df_day.reset_index()
ax = df_day.plot(kind='scatter', x="day", y='Basel Temperature')
# điều chỉnh độ chia nhỏ nhất của trục y và trục xc
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # mỗi 1 độ

plt.title('Temperature & Precipitation of Basel in April (2014 - 2025)')
plt.show()
##print (train_df[['day', 'Basel Temperature', 'Basel Precipitation Total']])

