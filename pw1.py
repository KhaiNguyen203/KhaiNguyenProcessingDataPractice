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



#tính trung bình dữ liệu của các giờ trong ngày
df_day = df.groupby('date')[['Basel Temperature', 'Basel Precipitation Total']].mean()
#Xóa các ô dữ liệu trống
df_day = df_day[['Basel Temperature', 'Basel Precipitation Total']].dropna()

# chia dữ liệu thành 2 phần để train và để test
df_train = df_day[:'2024-12-31']
df_test = df_day['2025-01-01':]

##vẽ biểu đổ 
# Chuyển 'date' từ index thành cột
df_day = df_day.reset_index()
ax = df_day.plot(kind='scatter', x="date", y='Basel Temperature')

# điều chỉnh độ chia nhỏ nhất của trục y và trục x
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # mỗi 1 độ

plt.title('Temperature & Precipitation of Basel in April (2014 - 2025)')

#*plt.show()
#*print (df_train[['date', 'Basel Temperature', 'Basel Precipitation Total']])

#rút trích dữ liệu từ tháng 4 của các năm trước 2024
df_train = df_train.reset_index()
df_test = df_test.reset_index()
df_april = df_train[df_train['date'].dt.month == 4]
temp_past_years = []
for i in range(2014, 2024):
    temp = df_april[df_april['date'].dt.year == i]
    temp_past_years.append(temp['Basel Temperature'])

#rút trích dữ liệu tháng 4 của năm 2024 để dán nhãn ở phía dưới
temp_in_2024 = []
temp_2024 = df_april[df_april['date'].dt.year == 2024]
temp_in_2024.append(temp_2024['Basel Temperature'])
'''
# Chuẩn hóa lại list thành array
training_data = np.array(temp_past_years)
lable_data = np.array(temp_in_2024)
'''
# Chuyển temp_past_years thành mảng 2D (số năm, số ngày)
training_data = np.array([np.array(temp) for temp in temp_past_years])  # shape (10, 30)

# Chuyển temp_in_2024 thành mảng 1D (số ngày trong tháng 4 của năm 2024)
lable_data = np.array(temp_in_2024).flatten()

print("Training data shape:", training_data.shape)
print("Label data shape:", lable_data.shape)
print(df_train)
'''
#tạo mô hình dự đoán
predictionModel = LinearRegression()
predictionModel.fit(training_data,lable_data)
'''
#*print (lable_data)