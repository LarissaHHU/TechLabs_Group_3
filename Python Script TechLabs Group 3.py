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



