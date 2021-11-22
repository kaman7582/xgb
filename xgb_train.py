from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.model_selection import train_test_split

all_data=read_csv("changsha_data.csv")

col_name=['H2','CO','CO2','C2H2','total']


def preprocess_data(dataset,history):
    data_x = []
    data_y = []
    for i in range(len(dataset)-history):
        data_x.append(dataset[i:i+history])
        data_y.append(dataset[i+history])
    return np.array(data_x),np.array(data_y)

for name in col_name:
    val = np.array(all_data[name])
    trainx,trainy = preprocess_data(val,5)
    X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = XGBRegressor(n_estimators=3000,
                     max_depth=10,
                     colsample_bytree=0.5, 
                     subsample=0.5, 
                     learning_rate = 0.01
                    )
    model.fit(X_train, y_train, verbose=True)
    model_name = "{}.model".format(name)
    model.save_model(model_name)

print("train finish")
'''
回归树个数：n_estimators=3000
每个回归树的深度：max_depth=10
生成树时进行的列采样：colsample_bytree=0.5
随机采样训练样本  : subsample':0.5
学习速率/步长0.0-1.0的超参数 每个树学习前一个树的残差的步长 : learning_rate=0.1

'''