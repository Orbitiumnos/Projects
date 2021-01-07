import numpy as np
import pandas as pd

from pandas import read_excel
from statsmodels.formula.api import ols
workbook = ('C:/Program1/Lab.xlsx')
sheetName = 'data'
data = read_excel(io = workbook, sheet_name = 'Лист1') #создали таблицу
cols = [c for c in data.columns if c.lower()[:1] != 'd'] #настраиваем фильтр, то есть все колонки с буквой d убираем
n = len(data.D1)

del_d = data[cols].drop(columns=['Residuals']) #убираем из массива, уже отфильтрованного, колонку residual

#проведение мультиколлинеарности с включением
def forward_selected(data, response):
   remaining = set(data.columns)
   remaining.remove(response)
   selected = []
   currentScore, bestNewScore = 0.0, 0.0
   while remaining and currentScore == bestNewScore:
       scoresWithCandidates = []
       for candidate in remaining:
           score = ols(formula="{} ~ {}".format(response, ' + '.join(selected + [candidate])), data=data).fit().rsquared_adj
           scoresWithCandidates.append((score, candidate))
           scoresWithCandidates.sort()
           bestNewScore, bestCandidate = scoresWithCandidates.pop()
           if currentScore < bestNewScore:
               remaining.remove(bestCandidate)
               selected.append(bestCandidate)
               currentScore = bestNewScore
               return ols(formula = "{} ~ {}".format(response, ' + '.join(sorted(selected))), data=data).fit()
forwardModel = forward_selected(del_d, 'Y') #команда для вызова процедуры мультиколлинеарности
print(forwardModel.summary()) #выводим результат
#-----------------------------------------------------------------------------------------------------------------------
del_d = data.drop(columns=['Residuals', 'Y']) #делаем новую модель, из которой убираем Y и Resid

formula = "{} ~ {}".format('Y', '+'.join(del_d.columns.tolist()))
model = ols(formula=formula, data=data).fit() #получаем модель в которую будут включены регресионные остатки
u = model.resid * model.resid #находим квадрат регресионных остатков (вектор)
RSS = u.sum() #получаем сумму квадратов регресионных остатков (суммируем весь вектор)
print('RSS =',RSS) #выводим RSS
#-----------------------------------------------------------------------------------------------------------------------
n_c = len(data[data.D1 != 0].D1) #здесь находим колличество строк в которых D1 = 1, то есть n1
del_d = data[cols].drop(columns=['Residuals', 'Y']) #очередная новая модель, убираем их нее все, кроме иксов
k = len(del_d.columns) #находим k
#-----------------------------------------------------------------------------------------------------------------------
data1 = data.sort_values(by=['D1'],ascending=False)#новая таблица, в которой убраны все строки после n1
data1 = data1.head(n_c)

del_n1 = data1.drop(columns=['Residuals', 'Y']) #все то же самое, что и выше, только для этого подмножества
formula1 = "{} ~ {}".format('Y', '+'.join(del_n1.columns.tolist()))
model1 = ols(formula=formula1, data=data1).fit()
u1 = model1.resid * model1.resid
RSS1 = u1.sum()
print('RSS1 =',RSS1)
n1 = model1.resid.count() #нашел n1, да опять, так надо
#-----------------------------------------------------------------------------------------------------------------------
data2 = data.sort_values(by=['D1'],ascending=False)#новая таблица, в которой убраны все строки после n1
data2 = data2.tail(n - n_c) #новая таблица, в которой убраны все строки до n1+1

del_n2 = data2.drop(columns=['Residuals', 'Y'])
formula2 = "{} ~ {}".format('Y', '+'.join(del_n2.columns.tolist()))
model2 = ols(formula=formula2, data=data2).fit()
u2 = model2.resid * model2.resid
RSS2 = u2.sum()
print('RSS2 =',RSS2)
n2 = model2.resid.count() #расчитал n2
#-----------------------------------------------------------------------------------------------------------------------
chow_nom = (RSS - (RSS1 + RSS2)) / (3 + 1) #считаем формулу, думаю тут все понятно
chow_denom = (RSS1 + RSS2) / (n1 + n2 - (2*k) - 2)
chow = chow_nom / chow_denom
print('Fрасч =',chow) #выводим f расчетное
#-----------------------------------------------------------------------------------------------------------------------
del_d = data.drop(columns=['Residuals']) #убираем из массива строку Resid перед мультиколлинеарностью
#проведение мультиколлинеарности
def forward_selected(data, response):
   remaining = set(data.columns)
   remaining.remove(response)
   selected = []
   currentScore, bestNewScore = 0.0, 0.0
   while remaining and currentScore == bestNewScore:
       scoresWithCandidates = []
       for candidate in remaining:
           score = ols(formula="{} ~ {}".format(response, ' + '.join(selected + [candidate])), data=data).fit().rsquared_adj
           scoresWithCandidates.append((score, candidate))
           scoresWithCandidates.sort()
           bestNewScore, bestCandidate = scoresWithCandidates.pop()
           if currentScore < bestNewScore:
               remaining.remove(bestCandidate)
               selected.append(bestCandidate)
               currentScore = bestNewScore
               return ols(formula = "{} ~ {}".format(response, ' + '.join(sorted(selected))), data=data).fit()
forwardModel = forward_selected(del_d, 'Y')
print(forwardModel.summary())