# -*- coding: utf-8 -*-

from settings import fname
from model.arimaModel import loadData
from model.arimaModel import selfRelatedPlot
#from model.arimaModel import whiteNoiseCheck
from model.arimaModel import stableCheck
def predictSales(fname,n=12,isVisiable=True):
    # 加载数据
    data = loadData(fname)
    # 画出自相关图
    selfRelatedPlot(data)
     # 对序列平稳性检测
    D_result = stableCheck(data)
    print('差分序列的ADF 检验结果为：', D_result)
    
    #对一阶差分后的序列做白噪声检验
    #print(u'差分序列的白噪声检验结果：',whiteNoiseCheck(data)) #返回统计量和 p 值


if __name__ == '__main__':
    # isVisiable可视化按钮
    result = predictSales(fname,12,isVisiable=True)