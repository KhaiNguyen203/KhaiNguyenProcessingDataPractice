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

#rút trích dữ liệu từ tháng 4 của các năm 2014-2023
df_train = df_train.reset_index()
df_april = df_train[df_train['date'].dt.month == 4]
temp_2014to2023 = []
for i in range(2014, 2024):
    temp = df_april[df_april['date'].dt.year == i]
    temp_2014to2023.append(temp['Basel Temperature'])

#rút trích dữ liệu từ tháng 4 của các năm 2015-2024 để dán nhãn
temp_2015to2024 = []
for i in range(2015, 2025):
    temp = df_april[df_april['date'].dt.year == i]
    temp_2015to2024.append(temp['Basel Temperature'])

#chuẩn hóa để phù hợp với phương thức fit
## Chuyển temp_2014to2023 thành mảng 2D (số năm, số ngày)
training_data = np.array([np.array(temp) for temp in temp_2014to2023])  # shape (10, 30)
## Chuyển temp_2015to2024 thành mảng 1D (số ngày trong tháng 4 của năm 2024)
label_data = np.array(temp_2015to2024)

'''
print("Training data shape:", training_data.shape)
print("Label data shape:", label_data.shape)
print(label_data)
'''

#tạo mô hình dự đoán
PModel = LinearRegression()
PModel.fit(training_data,label_data)

#! Phương thức values sẽ chuyển từ Series của pandas về array của numpy giúp chúng hoạt động được trong phương thức predict của scikitlearn
Temp_April_2024 = df_april[df_april['date'].dt.year == 2024]['Basel Temperature'].values

#! trong cấu trúc của reshape giúp biến mảng 1D (hiện tại là Temp_April_2024) biến thành mảng 2D (input_2024) 
##! số 1 biểu diễn cho số hàng của mảng 2D (chỉ có 1 hàng vì chỉ xét trong tháng 4 của năm 2024)
##! số 30 biêu diễn cho số cột của mảng 2D (30 ngày)
#phương thức predict yêu cầu 1 mảng 2D vì vậy cần chuyển từ 1D sang 2D
Temp_April_2024_reshape = Temp_April_2024.reshape(1, 30)

predicted_2025 = PModel.predict(Temp_April_2024_reshape)

# chuyển narray của predict_2025 thành data frame và thêm vào ngày giờ
dates_April_2025 = pd.date_range(start='2025-04-01', end='2025-04-30')
## chuyển predicted_2025 thành mảng 1D để không bị lỗi
predicted_2025 = predicted_2025.flatten()
## tạo ra hai cột thuộc data frame là date và predicted_temp
df_predicted_2025 = pd.DataFrame({'date': dates_April_2025, 'predicted_temp': predicted_2025})

# vẽ biểu đổ 
plt.title('Temperature & Precipitation of Basel in April (2014 - 2025)')
## Chuyển 'date' từ index thành cột
df_train = df_train.reset_index()

ax = df_train.plot(kind='scatter', x="date", y='Basel Temperature', color='#0077ff', label = 'Mesured temperature')
df_predicted_2025.plot(kind='scatter', x="date", y='predicted_temp', ax = ax, color='orange', label='Predicted temperature')

## điều chỉnh độ chia nhỏ nhất của trục y và trục x
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # mỗi 1 độ



plt.show()

#* print(df_train)