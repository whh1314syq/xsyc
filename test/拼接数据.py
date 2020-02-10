# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df1=pd.read_csv("../data/sales.csv",header=None)
df2=pd.read_csv("../data/sales1.csv",header=None)

df3=pd.DataFrame({"timestamp":df1.iloc[0:df2.shape[0],0],"sales":df2[1]})
df3.to_csv("../data/sales3.csv",index=None)