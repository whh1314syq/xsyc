# -*- coding: utf-8 -*-
#import pandas as pd
#from settings import fname
from model.arimaModel import loadData
from settings import fname
def exponential_smoothing(alpha, s):
    
     '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
  
     s_temp=[]
     s_temp.append(s[0])
     #s_temp = [0 for i in range(len(s))]
     #s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
     print(s_temp)
     for i in range(1, len(s)):
        s_temp.append(alpha * s[i-1] + (1 - alpha) * s_temp[i-1])
     return s_temp


def compute_single(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    return exponential_smoothing(alpha, s)

def compute_double(alpha, s):
    '''
    二次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回二次指数平滑模型参数a, b， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)

    a_double = [0 for i in range(len(s))]
    b_double = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_double[i] = 2 * s_single[i] - s_double[i]                    #计算二次指数平滑的a
        b_double[i] = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])  #计算二次指数平滑的b

    return a_double, b_double

def compute_triple(alpha, s):
    '''
    三次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回三次指数平滑模型参数a, b, c， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)
    s_triple = exponential_smoothing(alpha, s_double)

    a_triple = [0 for i in range(len(s))]
    b_triple = [0 for i in range(len(s))]
    c_triple = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])

    return a_triple, b_triple, c_triple

'''
def exponential_smoothing_3(alpha, s):
     
    三次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回三次指数平滑模型参数a, b, c， list
    
     s_single = exponential_smoothing(alpha, s)
     s_double = exponential_smoothing(alpha, s_single)
     s_triple = exponential_smoothing(alpha, s_double)
    
     a_triple = [0 for i in range(len(s))]
     b_triple = [0 for i in range(len(s))]
     c_triple = [0 for i in range(len(s))] 
     for i in range(len(s)):
        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])
     return a_triple, b_triple, c_triple
'''
def predict_value_with_exp_smoothing(alpha,s):
      a,b,c=compute_triple(alpha,s)
      s_temp=[]
      s_temp.append(a[0])
      for i in range(len(a)):
         s_temp.append(a[i]+b[i]+c[i])
      return s_temp
'''
def loadData(fname):
    
    导入数据
    :return:

    data = pd.read_excel(fname, index_col = 'date',header = 0)
    return data 
'''
if __name__ == "__main__":

    #alpha = 0.8
    #data = [i for i in range(100)]
     # 加载数据
    data = loadData(fname)
    print(predict_value_with_exp_smoothing(0.8,data['sales']))
    

    #sigle = compute_triple(alpha, data)

    #print(alpha * data[-1] + (1 - alpha) * sigle[-1])

