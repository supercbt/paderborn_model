
# 去除了复杂的加窗功能
# 增加Label标签

import pandas as pd
import numpy as np
import sys
import csv
from scipy import stats,signal

import pywt #导入PyWavelets

# 频域特征
from scipy.fftpack import fft, fftshift

# 时频域特征
import pywt
sampling_rate = 1024 #采样频率


argvs = sys.argv
path = argvs[1].split('=')[1]
opath = argvs[2].split('=')[1]

try:
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data_attribute = []
        result_out = []
        result_list = []
        list_para = []
        for i, rows in enumerate(reader):
            # label 存一下吧
            lb = np.array(rows)[-1]
            d = np.delete(np.array(rows), -1)
            if i == 0: 
                data_attribute.append('avg')  # 平均值
                data_attribute.append('std')  # 标准差
                data_attribute.append('var')  # 方差
                # data_attribute.append('skew')  # skew 负特征
                # data_attribute.append('kur') # 一个特征 负特征
                # data_attribute.append('ptp') # 又一个特征 无用特征
                # data_attribute.append('msa') # 方根幅值 负特征
                data_attribute.append('rms') # rms均方根
                data_attribute.append('ff') # fengfeng值
                data_attribute.append('cres') # 峰值因子
                data_attribute.append('clear') # 
                data_attribute.append('shape') # 波形因子
                # data_attribute.append('imp') # 脉冲指数 无用特征

                # data_attribute.append('fft') # fft均值
                # data_attribute.append('fft_p') # fft相位
                # data_attribute.append('fftshift')
                # data_attribute.append('fftpower')

                # 一套频域特征
                data_attribute.append('f_12') 
                # data_attribute.append('f_13') 
                # data_attribute.append('f_14')
                data_attribute.append('f_15')
                # data_attribute.append('f_16') 
                data_attribute.append('f_17') 
                data_attribute.append('f_18')
                data_attribute.append('f_19')
                # data_attribute.append('f_20') 
                data_attribute.append('f_21') 
                data_attribute.append('f_22')
                data_attribute.append('f_23')

                data_attribute.append('coeffs')
                # data_attribute.append('coeffs_var')
                # data_attribute.append('comsa')

                data_attribute.append('label')
            else:  # 跳过表头
                # 为每一横行增加特征
                # print(i)
                d = d.astype(float)
                # d = np.array(rows).astype(float)
                list_para.append(np.mean(d)) # avg
                list_para.append(np.std(d)) # std
                list_para.append(np.var(d)) # var
                # list_para.append(stats.skew(d)) # skew
                # list_para.append(stats.kurtosis(d)) # kur
                # list_para.append(d.ptp()) # ptp
                msa = ((np.mean(np.sqrt(np.abs(d)))))**2 
                # list_para.append(msa) # msa
                rms = np.sqrt((np.mean(d**2)))
                list_para.append(rms) # rms
                list_para.append(0.5*(np.max(d)-np.min(d))) # ff
                cres = np.max(np.abs(d))/rms
                list_para.append(cres) # cres
                clear = np.max(np.abs(d))/msa
                list_para.append(clear) # clear
                shape = (len(d) * rms)/(np.sum(np.abs(d)))
                list_para.append(shape) # shape
                imp = (np.max(np.abs(d)))/(np.mean(np.abs(d)))
                # list_para.append(imp) # imp

                d = np.array(d)
                L = len(d)
                # print('12')
                y = abs(np.fft.fft(d / L))[: int(L / 2)]
                # print('12')
                y[0] = 0
                x = np.fft.fftfreq(L, 1 / 25600)[: int(L / 2)]
                K = len(y)

                f_12 = np.mean(y) # fft均值
                # print('12')
                f_13 = np.var(y) # fft方差
                # f_14 = (np.sum((y - f_12)**3))/(K * ((np.sqrt(f_13))**3))
                f_15 = (np.sum((y - f_12)**4))/(K * ((f_13)**2))
                f_16 = (np.sum(x * y))/(np.sum(y))
                f_17 = np.sqrt((np.mean(((x- f_16)**2)*(y))))
                f_18 = np.sqrt((np.sum((x**2)*y))/(np.sum(y)))
                f_19 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y)))
                # f_20 = (np.sum((x**2)*y))/(np.sqrt((np.sum(y))*(np.sum((x**4)*y))))
                f_21 = f_17/f_16
                f_22 = (np.sum(((x - f_16)**3)*y))/(K * (f_17**3))
                f_23 = (np.sum(((x - f_16)**4)*y))/(K * (f_17**4))

                list_para.append(f_12)
                # list_para.append(f_13)
                # list_para.append(f_14)
                list_para.append(f_15)
                # list_para.append(f_16)
                list_para.append(f_17)
                list_para.append(f_18)
                list_para.append(f_19)
                # list_para.append(f_20)
                list_para.append(f_21)
                list_para.append(f_22)
                list_para.append(f_23)

                coeffs = pywt.wavedec(d, 'bior3.7', level = 5)
                co0= np.mean((coeffs[0]))
                # co1= np.std((coeffs[0]))
                # comsa = ((np.mean(np.sqrt(np.abs(coeffs[0])))))**2 
                list_para.append(co0)
                # list_para.append(co1)
                # list_para.append(comsa)


                # e_fft = np.abs(fft(np.array(d)))
                # list_para.append(np.mean(e_fft)) # fft均值

                # p_fft = np.angle(fft(np.array(d)))
                # list_para.append(np.mean(p_fft))

                # e_ffts = np.abs(fftshift(np.array(d)))
                # list_para.append(np.mean(e_ffts)) # fftshift均值

                # list_para.append(np.mean(np.square(e_fft)))

                # list_para.append(lb)

                result_list.append([])
                result_list[i - 1].extend(list_para)
                list_para = []
        result_out.append(data_attribute)

        

        for i, item in enumerate(result_list):
            result_out.append(item)
        # print(result_list)
        wrtocsv = pd.DataFrame(result_out)
        wrtocsv.to_csv(opath, index=False, header=False)

        path = opath  #特征提取后的csv文件路径
        df = pd.DataFrame(pd.read_csv(path))

except Exception as e:
    print(e)
