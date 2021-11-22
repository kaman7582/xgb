from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

all_data=read_csv("changsha_data.csv")

def preprocess_data(dataset,history):
    data_x = []
    data_y = []
    for i in range(len(dataset)-history):
        data_x.append(dataset[i:i+history])
        data_y.append(dataset[i+history])
    return np.array(data_x),np.array(data_y)

col_name=['H2','CO','CO2','C2H2','total']
total_cnt = len(col_name)
#val = list(all_data['total'])
i = 1
for name in col_name:
    val = np.array(all_data[name])
    testx,_ = preprocess_data(val,5)
    model_name = "{}.model".format(name)
    model= xgb.Booster(model_file=model_name)
    testx  = xgb.DMatrix(testx)
    test_result = model.predict(testx)
    total_len=len(val)
    test_len=(int)(total_len*0.2)
    test_start = total_len-test_len
    plt.subplot(5,1,i)
    plt.plot(val,'b',label=name)
    plt.plot(range(test_start,total_len),test_result[-test_len:], 'r', label='prediction')
    if(name =='C2H2'):
        print(test_result[-test_len:])
        print(val[-test_len:])
    plt.legend(loc="upper left",fontsize=8,shadow=True)
    plt.xticks([])
    max_val =max(np.max(val),1)
    plt.plot((test_start, test_start),(0,max_val), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    i += 1
    #plt.plot(val,color='blue')
    #plt.plot(range(total_len-test_len,total_len),test_result[-test_len:],color='red')
    #plt.show()

mon_local=[0,46,86,106,139,168]
mon_name=['2016-6-2','2017-1-2','2018-1-9','2019-1-2','2019-12-19','2021-1-2']
plt.xticks(mon_local, mon_name,rotation=45,fontsize=6)
#plt.legend(fontsize=3,shadow=True)
plt.savefig('predict.png',dpi=2048,format='png')
plt.show()
'''
model= xgb.Booster(model_file='total.model')


testx,_ = preprocess_data(val,5)
testx  = xgb.DMatrix(testx)
test_result = model.predict(testx)
plt.plot(val,color='blue')
xrange=range(5,len(val))
plt.plot(xrange,test_result,color='red')
plt.show()

'''