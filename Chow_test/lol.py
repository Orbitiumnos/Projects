import numpy as np
from pandas import read_excel
workbook = ('C:/Program1/Lab.xlsx')
sheetName = 'data'
data = read_excel(io = workbook, sheet_name = 'Лист1')
t = data.head(5)
# print(t)

# IndependentVariables = data.drop(columns=['D'])
del_d = data.drop(columns=['Residuals', 'Y'])
# print (del_d.describe())
# print(del_d)

from statsmodels.formula.api import ols
formula = "{} ~ {}".format('Y', '+'.join(del_d.columns.tolist()))
# print(formula)
model = ols(formula=formula, data=data).fit()
#print(model.summary())
u = model.resid * model.resid
RSS = u.sum()
print(RSS)

data1 = data.drop(index=[ 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,102,103,104,105,106,107,108,109,110,111,112, 113,114,115,116,117])
del_n1 = data1.drop(columns=['Residuals', 'Y'])
#print(del_n1)
from statsmodels.formula.api import ols
formula1 = "{} ~ {}".format('Y', '+'.join(del_n1.columns.tolist()))
#print(formula1)
model1 = ols(formula=formula1, data=data1).fit()
#print(model1.resid.head)
u1 = model1.resid * model1.resid
RSS1 = u1.sum()
print(RSS1)
n1 = model1.resid.count()

data2 = data.drop(index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
del_n2 = data1.drop(columns=['Residuals', 'Y'])
#print(del_n2)
from statsmodels.formula.api import ols
formula2 = "{} ~ {}".format('Y', '+'.join(del_n2.columns.tolist()))
#print(formula2)
model2 = ols(formula=formula2, data=data2).fit()
#print(model2.resid.head)
u2 = model2.resid * model2.resid
RSS2 = u2.sum()
print(RSS2)
n2 = model2.resid.count()

chow_nom = (RSS - (RSS1 + RSS2)) / 4
chow_denom = (RSS1 + RSS2) / (n1 + n2 - 6 - 2)
chow = chow_nom / chow_denom
print(chow)