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
df_train = df[['Pclass', 'Sex']]
list_train = []
for i in range (0, 891):
    tmp = []
    #item giúp chuyển từ kiểu dữ liệu của numpy về python
    tmp.append(df_train['Pclass'].loc[i].item())
    if(df_train['Sex'].loc[i] == 'male'):
        tmp.append(1)
    else:
        tmp.append(0)
    list_train.append(tmp)
list_label =[]
for i in range(0, 891):
    list_label.append(df['Survived'].loc[i])

ar_train = np.array(list_train)
ar_label = np.array(list_label)

df_test = pd.read_csv("test.csv")
df_test = df_test[['PassengerId', 'Pclass', 'Sex']]
list_test = []
for i in range (1, 418):
    tmp = []
    #item giúp chuyển từ kiểu dữ liệu của numpy về python
    tmp.append(df_test['Pclass'].loc[i].item())
    if(df_test['Sex'].loc[i] == 'male'):
        tmp.append(1)
    else:
        tmp.append(0)
    list_test.append(tmp)
ar_test = np.array(list_test)

model = LinearRegression()
model.fit(ar_train, ar_label)

survived_prediction = model.predict(ar_test)
survived_prediction = survived_prediction.flatten()
#transfer PassengerId col from df_test to array 1D
transfer_col_PassengerId = []
for i in range(1, 418):
    transfer_col_PassengerId.append(df_test['PassengerId'].loc[i].item())
    if(survived_prediction[i-1] >= 0.5): survived_prediction[i-1] = int(1)
    else: survived_prediction[i-1] = int(0)
df_result = pd.DataFrame({"PassengerId": transfer_col_PassengerId, "Survived": survived_prediction})
print (df_result)
#* df_result.to_csv('result.csv', index=False)

###
###
###