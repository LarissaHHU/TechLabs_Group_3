#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#import the seperate excel files from my research into Python
QOLI = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Quality of Life Index.xlsx')
HE = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Health Expenditure.xlsx')
LE = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Life Expectancy.xlsx')
HS = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Healthcare System.xlsx')

print(QOLI)
print(HE)
print(LE)
print(HS)

#explanation of abbreviations for going further:
    #QOLI: Quality of Life Index
    #HI: Health Expenditure
    #LE: Life Expectancy
    #HS: Healthcare System

#merging all datasets into one
QOLI_df = pd.merge(QOLI, HE, LE, HS)

print(QOLI_df)


#pandas melt for getting data structured in the way of: country, index_name_year, index_value

QOLI_DF_melt = pd.melt(QOLI_DF, id_vars=['Country'], value_vars=['QOLI_2012', 'QOLI_2013', 'QOLI_2014', 'QOLI_2015', 'QOLI_2016', 'QOLI_2017', 'QOLI_2018', 'QOLI_2019', 'QOLI_2020', 'QOLI_2021', 'QOLI_2022', 'QOLI_2023'], value_name='value', col_level=None, ignore_index=True)

print(QOLI_DF_melt)

#Export the DataFrame to an Excel file
QOLI_DF_melt.to_excel('QOLI2015_2023.xlsx', index=False)

with pd.ExcelWriter('QOLI2015_2023.xlsx') as excel_writer:
    #QOLI_DF_melt.to_excel(excel_writer, sheet_name='Sheet1', index=False)

import os

path = r'C:\Users\Pomrehn\Desktop\TechLabs'
csv_files = [os.path.join(path+"\\", file) for file in os.listdir(path) if file.endswith('.csv')]

QOLI_DF_melt = [pd.read_csv(d) for d in csv_files]

QOLI_DF_melt.to_csv("QOLI2015_2023.csv")


