import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Đây là module để điều chỉnh các mốc chia (ticks) trên trục (x hoặc y).
import matplotlib.ticker as ticker
# Đây là module chuyên để xử lý các trục thời gian (ngày/tháng/năm).
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression

BaTemp = 'Basel Temperature'
BaPrec = 'Basel Precipitation Total'

df = pd.read_csv('2014-2025_WeatherForecastDataInBasel.csv')

# chuyển dạng timestamp sang dạng yyyy-mm-dd
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M')

#Gộp phần giờ của mẫu số liệu chỉ lấy theo ngày
df['date'] = pd.to_datetime(df['timestamp'].dt.date) # hoặc .dt.to_period('D') cũng được

df = df.sort_values(by="date")

# set index là ngày ở đây để có thể slice mẫu dữ liệu theo ngày
df = df.set_index('date')

# chia dữ liệu thành 2 phần để train và để test
df_train = df[:'2024-12-31']
df_test = df['2025-01-01':]

#tính trung bình dữ liệu của các giờ trong ngày
df_day = df_train.groupby('date')[['Basel Temperature', 'Basel Precipitation Total']].mean()
#Xóa các ô dữ liệu trống
df_day = df_day[['Basel Temperature', 'Basel Precipitation Total']].dropna()

##vẽ biểu đổ 
# Chuyển 'date' từ index thành cột
df_day = df_day.reset_index()
ax = df_day.plot(kind='scatter', x="date", y='Basel Temperature')

# điều chỉnh độ chia nhỏ nhất của trục y và trục x
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # mỗi 1 độ

plt.title('Temperature & Precipitation of Basel in April (2014 - 2025)')
## 

#*plt.show()
#*print (df_train[['date', 'Basel Temperature', 'Basel Precipitation Total']])

#rút trích dữ liệu từ tháng 4 của từng năm
df_train = df_train.reset_index()
df_april = df_train[df_train['date'].dt.month == 4]
temp_past_years = []
for i in range(2014, 2024):
    temp = df_april[df_april['date'].dt.year == i]
    temp_past_years.append(temp['Basel Temperature'])
#rút trích dữ liệu tháng 4 của năm 2024 để dán nhãn ở phía dưới
temp_in_2024 = []
temp_2024 = df_april[df_april['date'].dt.year == 2024]
temp_in_2024.append(temp_2024['Basel Temperature'])

#dữ liệu cho máy học
training_data = [[]]
for i in range (len(temp_past_years)):
    training_data.append(temp_past_years[i])

# dữ liệu gán nhãn
lable_data = []
lable_data.append(temp_in_2024)

#tạo mô hình dự đoán
predictionModel = LinearRegression()
predictionModel.fit(training_data,lable_data)
#*print (lable_data)