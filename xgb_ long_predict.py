from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from queue import Queue
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

input_queue = Queue(maxsize=5)

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
    if(name == 'C2H2'):
        val = np.array(all_data[name])
        testx,_ = preprocess_data(val,5)
        model_name = "{}.model".format(name)
        model= xgb.Booster(model_file=model_name)
        test_x  = xgb.DMatrix(testx)
        test_result = model.predict(test_x)
        
        #first_input = testx[0]
        #first_input = first_input.reshape(1,-1)
        #first_input_x = xgb.DMatrix(first_input)
        #
        #first_input = testx[0].reshape(1,-1)
        predict_result = []
        testx = testx.reshape(-1)
        for i in testx[-30:-25]:
            input_queue.put(i)
        print(input_queue.queue)
        for i in range(30):
            predict_data = np.array(input_queue.queue,dtype=float).reshape(1,-1)
            predict_data = xgb.DMatrix(predict_data)
            res = model.predict(predict_data)
            predict_result.append(float(res))
            input_queue.get()
            input_queue.put(res)
        
        print(predict_result)
        plt.plot(val[-30:],'b')
        len_rg = range(len(val),len(val)+30)
        plt.plot(predict_result,'r')
        plt.show()
        y_test = predict_result
        y_predict = val[-30:]
        print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
        print("mean_squared_error:", mean_squared_error(y_test, y_predict))
        print("rmse:", sqrt(mean_squared_error(y_test, y_predict)))
        print("r2 score:", r2_score(y_test, y_predict))

        '''
        y_test = predict_result
        #y_predict = test_result[0:30]
        y_predict = val[5:35]
        print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
        print("mean_squared_error:", mean_squared_error(y_test, y_predict))
        print("rmse:", sqrt(mean_squared_error(y_test, y_predict)))
        print("r2 score:", r2_score(y_test, y_predict))
        plt.plot(y_test,'b',label="predict")
        plt.plot(y_predict,'r',label="raw")
        plt.legend(fontsize=8,shadow=True)
        plt.show()
        '''

    '''
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