#!/usr/bin/env python
# coding: utf-8

# In[1]:

#1. DATA PREP
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

QOLI_DF_melt.to_csv(r'C:\Users\Pomrehn\Desktop\TechLabs\QOLI2015_2023.xlsx')

#2. DESCRIPTIVE STATISTICS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

QOLI_DF.shape
#97, 139

QOLI_DF.head()

QOLI_DF.info()

QOLI_DF.isnull().sum()

QOLI_DF.describe()

#QOLI_DF.describe(include='all')

print(QOLI_DF['QOLI_2012'].mean)
print(QOLI_DF['QOLI_2013'].mean)
QOLI_DF['QOLI_2014'].mean
QOLI_DF['QOLI_2015'].mean
QOLI_DF['QOLI_2016'].mean
QOLI_DF['QOLI_2017'].mean
QOLI_DF['QOLI_2018'].mean
QOLI_DF['QOLI_2019'].mean
QOLI_DF['QOLI_2020'].mean
QOLI_DF['QOLI_2021'].mean
QOLI_DF['QOLI_2022'].mean
QOLI_DF['QOLI_2023'].mean

mean = QOLI_DF['QOLI_2012'].mean()

plt.boxplot(QOLI_DF['QOLI_2012'])
plt.show()
#Why does the boxplot won't work? Because of the description of the axes?

#Skweness
QOLI_DF['QOLI_2012'].skew()

#3. ANOVA

#ANOVA to see if there is differences between groups for the countries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#current dataset
QOLI_DF = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Dataset Quality of Life Index_21082023.xlsx')

#country list for groups
countries_of_interest = [
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahrain",
    "Bangladesh", "Belarus", "Belgium", "Bolivia", "Bosnia & Herzegovina",
    "Brazil", "Bulgaria", "Cambodia", "Canada", "Chile", "China", "Colombia",
    "Costa Rica", "Croatia", "Cyprus", "Czech Republic", "Denmark",
    "Dominican Republic", "Ecuador", "Egypt", "Estonia", "Finland", "France",
    "Georgia", "Germany", "Greece", "Hong Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Ireland", "Israel", "Italy", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kuwait", "Latvia", "Lebanon", "Lithuania",
    "Luxembourg", "Malaysia", "Malta", "Mexico", "Moldova", "Mongolia",
    "Morocco", "Netherlands", "New Zealand", "Nigeria", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Panama", "Peru", "Philippines", "Poland",
    "Portugal", "Puerto Rico", "Qatar", "Romania", "Russia", "Saudi Arabia",
    "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa", "South Korea",
    "Spain", "Sri Lanka", "Sweden", "Switzerland", "Taiwan", "Thailand", "Tunisia",
    "Turkey", "Turkmenistan", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Venezuela", "Vietnam"
]

#empty list to store the data for each country
country_data = []

#extract data for each country and store it in the 'country_data' list
for country in countries_of_interest:
    country_data.append(QOLI_DF[country])

#perform one-way ANOVA analysis
f_statistic, p_value = stats.f_oneway(*country_data)

#results
print("F-statistic:", f_statistic)
print("p-value:", p_value)

#interpret the results based on the p-value
alpha = 0.05  # Set your significance level
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences between at least two groups.")
else:
    print("Fail to reject the null hypothesis: There are no significant differences between the groups.")
    
#results:
#F-statistic: nan
#p-value: nan
#Fail to reject the null hypothesis: There are no significant differences between the groups.

#ANOVA to see if there is differences between groups for the years
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#current dataset
QOLI_DF = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Dataset Quality of Life Index_21082023.xlsx')

import pandas as pd
from scipy.stats import f_oneway

# Assuming QOLI_DF is your DataFrame containing the data

# Define a list of the columns (variables) you want to analyze
columns_of_interest = [
    'QOLI_2012', 'QOLI_2013', 'QOLI_2014', 'QOLI_2015', 'QOLI_2016', 'QOLI_2017',
    'QOLI_2018', 'QOLI_2019', 'QOLI_2020', 'QOLI_2021', 'QOLI_2022', 'QOLI_2023'
]

# Create an empty list to store the data for each variable
variable_data = []

# Extract data for each variable and store it in the 'variable_data' list
for column in columns_of_interest:
    variable_data.append(QOLI_DF[column])

# Perform one-way ANOVA analysis using f_oneway from scipy.stats
f_statistic, p_value = f_oneway(*variable_data)

# Print the results
print("F-statistic:", f_statistic)
print("p-value:", p_value)

# Interpret the results based on the p-value
alpha = 0.05  # Set your significance level
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences between the variables.")
else:
    print("Fail to reject the null hypothesis: There are no significant differences between the variables.")

#results:
#F-statistic: nan
#p-value: nan
#Fail to reject the null hypothesis: There are no significant differences between the variables.

import pandas as pd

QOLI_DF = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Dataset Quality of Life Index_21082023.xlsx')

#dummy for country
encoded_df = pd.get_dummies(QOLI_DF, columns=['Country'], drop_first=True)

#linear regression
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

#4. LINEAR REGRESSION

#perform multiple linear regression
QOLI_DF = pd.read_excel(r'C:\Users\Pomrehn\Desktop\TechLabs\Dataset Quality of Life Index_21082023.xlsx')

#replace missing values with 0
QOLI_DF.fillna(0, inplace=True)

#Check the data type of the "Country" column
print(QOLI_DF['Country'].dtype)

# Ensure that the "Country" column is of string data type
QOLI_DF['Country'] = QOLI_DF['Country'].astype(str)

#Encode 'Country' as numeric using Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
QOLI_DF['Country'] = label_encoder.fit_transform(QOLI_DF['Country'])

# Define the list of independent variables
independent_vars = [
    'QOLI_2012', 'PPI_2012', 'SI_2012', 'HCI_2012', 'COLI_2012', 'PPTIR_2012',
    'TCTI_2012', 'PI_2012',
    'QOLI_2013', 'PPI_2013', 'SI_2013', 'HCI_2013', 'COLI_2013', 'PPTIR_2013',
    'TCTI_2013', 'PI_2013',
    'QOLI_2014', 'PPI_2014', 'SI_2014', 'HCI_2014', 'COLI_2014', 'PPTIR_2014',
    'TCTI_2014', 'PI_2014',
    'QOLI_2015', 'PPI_2015', 'SI_2015', 'HCI_2015', 'COLI_2015', 'PPTIR_2015',
    'TCTI_2015', 'PI_2015',
    'QOLI_2016', 'PPI_2016', 'SI_2016', 'HCI_2016', 'COLI_2016', 'PPTIR_2016',
    'TCTI_2016', 'PI_2016', 'CI_2016',
    'QOLI_2017', 'PPI_2017', 'SI_2017', 'HCI_2017', 'COLI_2017', 'PPTIR_2017',
    'TCTI_2017', 'PI_2017', 'CI_2017',
    'QOLI_2018', 'PPI_2018', 'SI_2018', 'HCI_2018', 'COLI_2018', 'PPTIR_2018',
    'TCTI_2018', 'PI_2018', 'CI_2018',
    'QOLI_2019', 'PPI_2019', 'SI_2019', 'HCI_2019', 'COLI_2019', 'PPTIR_2019',
    'TCTI_2019', 'PI_2019', 'CI_2019',
    'QOLI_2020', 'PPI_2020', 'SI_2020', 'HCI_2020', 'COLI_2020', 'PPTIR_2020',
    'TCTI_2020', 'PI_2020', 'CI_2020',
    'QOLI_2021', 'PPI_2021', 'SI_2021', 'HCI_2021', 'COLI_2021', 'PPTIR_2021',
    'TCTI_2021', 'PI_2021', 'CI_2021',
    'QOLI_2022', 'PPI_2022', 'SI_2022', 'HCI_2022', 'COLI_2022', 'PPTIR_2022',
    'TCTI_2022', 'PI_2022', 'CI_2022',
    'QOLI_2023', 'PPI_2023', 'SI_2023', 'HCI_2023', 'COLI_2023', 'PPTIR_2023',
    'TCTI_2023', 'PI_2023', 'CI_2023'
]

#Encode 'Country' as numeric using Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
QOLI_DF['Country'] = label_encoder.fit_transform(QOLI_DF['Country'])

# Add a constant term to the independent variables (intercept)
X = sm.add_constant(QOLI_DF[independent_vars])

# Define the dependent variable
y = QOLI_DF['Country']

#Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())
