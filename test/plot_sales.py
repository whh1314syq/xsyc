# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
df=pd.read_csv("../data/sales3.csv",header=None)
from pyecharts import Line

line = Line("空调销售预测")
line.add("实际销量结果", df[0],df[1],line_color='black',line_width=2)


line.render('../data/新销量.html')