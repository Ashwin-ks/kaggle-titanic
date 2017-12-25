# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
os.chdir('D:\\Windows.old\\Users\\Public\\Documents\\kaggle\\kaggle-titanic\\data')
trndata=pd.read_csv('train.csv')
print(trndata.describe())
print(trndata.head())