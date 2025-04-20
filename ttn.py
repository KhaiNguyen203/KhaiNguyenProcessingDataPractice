import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")
'''
df.plot(kind = 'scatter', x='PassengerId', y="Survived")
plt.show()
'''
df_label = df['Survived']
df_train = df['Sex']

list_train = []
change_to_list_2D = []
for i in range (0, 891):
    tmp = df['Sex'].loc[i]
    if(tmp == 'male'): list_train.append(1)
    else: list_train.append(0)

list_label =[]
for i in range(0, 891):
    list_label.append(df['Survived'].loc[i])
df_train = np.array(list_train)
df_train = df_train.reshape(1, 891)
df_label = np.array(list_label)
df_label = df_label.reshape(1, 891)
#*print(df_train)
#*print (df_train.shape)
#*print (df_label.shape)

model = LinearRegression()
model.fit(df_train, df_label)

