#!/usr/bin/python3
# __*__ coding: utf-8 __*__

'''
@Author: syq
@Os：Windows 10 x64
@Software: PY PyCharm 
@File: sales.py
@Time: 2020/1/1 18:53
'''

#from model.smooth     import predict_value_with_exp_smoothing
#from model.arimaModel import diffData1
from model.arimaModel import loadData
from model.arimaModel import diffData
from model.arimaModel import sequencPlot
from model.arimaModel import sequencePlot
from model.arimaModel import selfRelatedPlot
from model.arimaModel import partialRelatedPlot
from model.arimaModel import stableCheck
from model.arimaModel import selectArgsForModel
from model.arimaModel import bulidModel
from model.arimaModel import predict
from model.arimaModel import roundResult
#from model.arimaModel import smooth
from model.arimaModel import whiteNoiseCheck
from settings import fname




def predictSales(fname,n=12,isVisiable=False):
    '''
    程序执行的入口
    :param fname:输入文件
    :param n:预测的点个数
    :param isVisiable:是否可视化
    :return:6个点就是两个月，每月分上中下旬三个点
    '''
    # 加载数据
    data = loadData(fname)
    #print(data)
    #dataset=predict_value_with_exp_smoothing(alpha=0.2,s=data['sales'])
    
    
 
    # 对序列差分处理
    D_data = diffData(data)
    
    #print('二阶差分数据：', D_data)
    if isVisiable:
       
        #画出原始数据时序图
        sequencPlot(data)
         #平滑指数时序图
        #smooth(dataset)
        # 画出差分后的时序图
        sequencePlot(D_data)
        # 画出自相关图
        selfRelatedPlot(D_data)
        # 画出偏相关图
        partialRelatedPlot(D_data)
    # 对差分序列平稳性检测
    D_result = stableCheck(D_data)
    print('差分序列的ADF 检验结果为：', D_result)
    #对一阶差分后的序列做白噪声检验
    print(u'差分序列的白噪声检验结果：',whiteNoiseCheck(D_data)) #返回统计量和 p 值
    
    # 对模型进行定阶
    p,q = selectArgsForModel(D_data)
    print('BIC 最小的p值 和 q 值：%s,%s' %(p,q))
    # 建立模型
    model = bulidModel(data,p,q)
    # 进行销量预测
    result = predict(model,n).tolist()
    # 对结果进行取整处理
    result = roundResult(result)
    print('预测未来n个点的销量为：',result)
    return result

if __name__ == '__main__':
    # isVisiable可视化按钮
    result = predictSales(fname,12,isVisiable=False)