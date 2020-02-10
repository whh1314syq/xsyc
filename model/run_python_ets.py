import numpy as np
import pandas as pd
from xl_HoltWinters import holtWinters
from pyecharts import Line

df=pd.read_csv("../data/sales.csv",header=None)
months=15
df.columns=['timestamp','sales']
columns_pre=df.timestamp.values[-months:]
df_pre=pd.DataFrame(np.zeros((months,months)))
df_pre.columns=columns_pre
df_pre["add1"]=0
df_pre['add2']=0
df_pre["add3"]=0

for i in range(df.shape[0]-months,df.shape[0]):
    tsA=df.iloc[0:i,1]
    
    result_ets=holtWinters(tsA, 4, 2, 4, mtype = 'additive')
    #result_ets=holtWinters(tsA, 12, 4, 4, mtype = 'other')
    result_ets2=np.round(list(result_ets['predicted']),2)
    print(i,tsA,result_ets2)
    
    begin_num=(i+months-df.shape[0])
    try:
        df_pre.iloc[begin_num,begin_num:begin_num+4]=result_ets2
        #df_pre.iloc[begin_num,begin_num:begin_num+4]=[1,1,1,1]
    except Exception as e:
        print('报错')
    
    
df_plot=df_pre.iloc[:,3:-3]
list_pre1=[]
list_pre2=[]
list_pre3=[]
list_pre4=[]
for j in range(0,df_plot.shape[1]):
    list_pre1.append(df_plot.iloc[j,j])
    list_pre2.append(df_plot.iloc[j+1,j])
    list_pre3.append(df_plot.iloc[j+2,j])
    list_pre4.append(df_plot.iloc[j+3,j])

        
date_all=df.timestamp[df.shape[0]-months+3:]
sales_real= df.sales[df.shape[0]-months+3:]

date_pre=df_plot.columns
sales1=list_pre1
sales2=list_pre2
sales3=list_pre3
sales4=list_pre4

line = Line("空调销售预测", title_pos='right', 
            width=1400, height=700, title_color='red',title_text_size=10)
line.add("实际销量结果", date_all, sales_real,line_color='black',line_width=2)

line.add("第一次预测结果", date_pre, sales1,line_type='dashed',line_color='green')
line.add("第二次预测结果", date_pre, sales2,line_type='dashed',line_color='green')
line.add("第三次预测结果", date_pre, sales3,line_type='dashed',line_color='green')
line.add("第四次预测结果", date_pre, sales4,line_type='dashed',line_color='green')
#line.add("python版arima预测结果",date_pre,sales_arima_python,line_type='dashed',line_color='red',line_width=2)
#line.add("商家B", attr, v2, is_smooth=True,mark_line=["max", "average"])
line.render('../data/python_ets.html')
'''
results1 = holtWinters(tsA, 12, 4, 4, mtype = 'additive')
results2 = holtWinters(tsA, 12, 4, 4, mtype = 'multiplicative')

print("TUNING: ", results1['alpha'], results1['beta'], results1['gamma'], results1['MSD'])
print("FINAL PARAMETERS: ", results1['params'])
print("PREDICTED VALUES: ", results1['predicted'])
'''
