#!/usr/bin/env python
# coding: utf-8

# ### Структура типичной нейронной сети Кохонена показана ниже: (во входном слое - 5 нейронов)

# #### ![image.png](attachment:image.png)
# 

# #### Импорт библиотек 

# In[1]:


import numpy as np 
import pandas as pd
import math
import time
from random import randint as rand


# #### Затем мы загружаем набор данных 

# In[2]:


# путь к файлу с данными
path = r"C:\Users\Николай\Desktop\Питон_нейронка\Data_for_neural_network_2.xlsx"
xl = pd.ExcelFile(path)
# читаем данные с нужного листа
dataset = pd.read_excel(xl, 'Data2')

# создаем writer для выгрузки результатов
writer = pd.ExcelWriter("Data_output_1.xlsx",engine = 'xlsxwriter')
# добавляем в него исходные данные
dataset.to_excel(writer, sheet_name = 'Data')

dataset.head(7) # проверяем , правильно ли считалось 


# #### Нормализуем данные 

# In[3]:


def normilize(df):
    features = df.columns[1:-1] #список признаков
    new_dict = {} #будущий датафрем с нормализованными данными
    for feature in features:
        vars()["list" + str(feature)] = [] #создание списка для каж
        for element in df[feature]:
            element = (element-min(df[feature]))/(max(df[feature])-min(df[feature])) #нормализация каждого столбца(признака)
            vars()["list" + str(feature)].append(element)
        new_dict[feature] = vars()["list" + str(feature)]
    return pd.DataFrame(new_dict)


# In[4]:


normilize_data = normilize(dataset) #нормализованные данные
normilize_data["Class"] = dataset["Class"]
normilize_data.head()


# In[5]:


# добавляем нормализованные данные в файл
normilize_data.to_excel(writer, sheet_name = 'Normalize')


# ### Делаем разделение данных на те, на которых будет проводиться кластеризация и те, что будут использоваться при классификации.

# In[6]:


def separation (data, percent):
    data = data.copy()
    types = [0]*len(data)
    for cl in data.Class.unique():
        count = int(len(data[data.Class == cl]) * (percent/100))
        print(cl, " ",count, " / ", len(data[data.Class == cl]))
        j = 0
        for i in data[data.Class == cl].index:
            if (rand(0,1) == 1):
                if (j < count):
                    types[i] = "Test"
                    j += 1
                    continue
            types[i] = "Train"
    data["Type"] = types
    return data


# In[7]:


# задаем процент количества записей для классификации
percent_of_classification = 20


# In[8]:


normilize_data = separation(normilize_data, percent_of_classification)
X_train = np.array(normilize_data[normilize_data.Type == "Train"].drop(columns=["Class", "Type"]))
X_test = np.array(normilize_data[normilize_data.Type == "Test"].drop(columns=["Class", "Type"]))
print(X_test)


# # Создаем классы нашей нейронной сети

# ## Класс Нейрона

# In[9]:


class Neuron():
    
    # Наш конструктор, где создаются (и инициализируются) 
    # все необходимые переменные
    def __init__(self):
        # вектор весов
        self.__W = []
        
    # Функция инициализации вектора весов нейрона 
    def iniciate_W(self, count_of_element, count_of_parameters):
        self.__W = self.__get_weights(count_of_element, count_of_parameters)

    ## Функция создания вектора весов нейрона
    def __get_weights(self, count_of_element, count_of_parameters):
        list_of_weghts = []
        for i in range(count_of_parameters):
            w = self.__get_weight(count_of_element)
            list_of_weghts.append(w)
        return list_of_weghts
    
    ## Функция получения случайного веса в диапазоне 
    ## от 0 до 1 
    def __get_weight (self, count_of_element):
        y  =  np.random.random ()  * ( 2.0  /  np .sqrt(count_of_element)) 
        return   0.5  - ( 1  /  np.sqrt(count_of_element))  +  y
    
    # Функция возврата вектора весов
    def get_W(self):
        return self.__W
    
    # Функция коррекции вектора весов, получающая на вход
    # вектор х входящего элемента и коэффициент коррекции
    def correct_W(self, coef, x):
        w = self.__W
        for i in range(len(w)):
            w[i] = w[i] + coef*(x[i] - w[i])
        self._W = w


# ## Класс сети Кохонена

# In[10]:


class NNK_NRV():
    
    def __init__(self, lambd = 0.7, alpha = 0.005, delta = None, 
                 count_of_clusters = 3, count_of_era=100):
        # λ coefficient
        self.__lambd = lambd
        
        # а coefficient
        self.__alpha = alpha
        
        # count of clusters
        self.count_of_clusters = count_of_clusters
        
        # count of nearest clusters to learn
        if delta is None:
            self.__delta = count_of_clusters-1
        else:
            self.__delta = delta
        
        # count of era for learn
        self.__count_of_era = count_of_era
        
        ## paraneters to learn
        self.count_of_element = None                  
        self.count_of_parameters = None
        
        self.__neurons = self.__get_neurons(count_of_clusters)
        
    # Функция обучения нейронной сети
    def fit(self, data):
        # получение числа элементов и количества атрибутов(параметров) для каждого элемента
        self.count_of_element, self.count_of_parameters = self.__get_count(data)
        
        # инициализация весов на нейронах
        self.__iniciate_of_neurons()
        
        lambd = self.__lambd
        
        # реализуем условие остановки обучения
        era = 0
        while (lambd >= 0) and (era < self.__count_of_era):
            # пробег по всем элементам полученных данных
            for i in range(len(data)):
                x = data[i]
                ind_of_win = self.__search_of_closest(x)
                self.__modifying_W(era, lambd, ind_of_win, x)
            
            # коррекция скорости обучения
            lambd = self.__delta_lambda(era, lambd, self.__alpha)
            
            # подсчет эры
            era += 1
            
    # Функция определения класса для элемента
    def define(self, data):
        result_list = []
        for i in range(len(data)):
            num_of_class =  self.__search_of_closest(data[i])
            result_list.append(num_of_class)
        return result_list
    
    ## Функция изменения весов нейоронов
    def __modifying_W(self, era, lambd, i_o_w, x):
        neuro = self.__neurons
        for ind in range(len(neuro)):
            neuro[ind].correct_W(x=x, 
                                 coef=self.__coef_of_learn(era, i_o_w, ind)*lambd)
    
    ## Функция изменения скорости обучения
    def __delta_lambda(self, era, lambd, alpha):
        e = math.exp(-era/alpha)
        return lambd*e
    
    ## Функция получения коэффициента обучения
    def __coef_of_learn(self, era, i_o_w, n_n_i):
        delta = 0.7
        d = self.__distance(self.__neurons[i_o_w].get_W(), 
                       self.__neurons[n_n_i].get_W())
        delta = delta/(1 + era/self.__count_of_era)
        hdt = 0
        if d <= delta:
            hdt = math.exp(-(d**2) / (2*(delta**2)))
        return hdt
        
    ## Функция сравнения входного вектора и весов нейронов для нахождения ближайшего
    def __search_of_closest(self, x):
        min_dist = None
        ind_of_winner = None
        for i in range(len(self.__neurons)):
            neuron_w = self.__neurons[i].get_W()
            dist = self.__distance(x, neuron_w)
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                ind_of_winner = i
        return ind_of_winner
    
    ## Функция вычисления евклидова растояния между двумя векторами
    def __distance(self, x, y):
        r = 0
        for i in range(len(x)):
            r = r + ((x[i] - y[i])**2)     #сумма квадратов расстояний
        r = np.sqrt(r)
        return r  
    
    ## Функция получения числа элементов и количества атрибутов(параметров) для каждого элемента
    def __get_count(self, data):
        return (len(data), len(data[0]))
    
    ## Функция для создания массива(слоя) нейронов
    def __get_neurons(self, count_of_clusters):
        list_of_clasters = []
        for i in range(count_of_clusters):
            list_of_clasters.append(Neuron())
        return list_of_clasters
    
    ## Функция инициализации весов на нейронах
    def __iniciate_of_neurons(self):
        for neuron in self.__neurons:
            neuron.iniciate_W(self.count_of_element, self.count_of_parameters)


# ## Класс оболочка для нейронной сети для нахождения оптимальных параметров lambda (скорость обучения), alpha (изменение скорости обучения) и delta (количество ближайших соседей для обучения)

# In[11]:


class auto_generate_const():
    
    def __init__(self):
        # partitioning quality functional
        ## функционал качества разбиения
        self.__pqf = None
        
        # время начала подбора параметров
        self.__time = None
        
        # итоговая нейронная сеть
        self.__best_NN = None
      
    # Функция обучения (подбора оптимальных параметров и 
    # последующего обучения нейронной сети на них)
    def fit(self, data, count_of_clusters=3, max_lambd=0.9):
        # Подбираем оптимальные параметры для нейронной сети
        res = self.__generate(data, count_of_clusters, max_lambd)
        
        # записываем функционал качества разбиения
        self.__pqf = res[0]
        
        # Обучаем на полученных параметрах итоговую нейронную сеть
        self.__best_NN = NNK_NRV(lambd=res[2], alpha=res[3], delta=res[1], 
                          count_of_clusters=count_of_clusters)
        self.__best_NN.fit(data)
        
        # Сообщаем о завершении и итоговое время выполнения
        print("fit comlete. Time of work: %.2f"%(time.time() - self.__time))
    
    # Функция для возврата итоговой нейронной сети
    def get_neuron_network(self):
        return self.__best_NN
    
    # Функция для возврата функционал качества разбиения
    def get_functional(self):
        return self.__pqf
    
    def colculate_functional(self, count_of_clusters, class_ar, data):
        return self.__fuctional_amount(self.__divisoin(count_of_clusters, class_ar, data))
    
    # Функция подбора оптимальных параметров
    def __generate(self, data, count_of_clusters=3, max_lambd=0.9):
        # записываем время начала
        self.__time = time.time()
        
        # записываем начальные параметры
        delta = count_of_clusters - 1
        lambd = max_lambd
        
        # массив результатов, состав строк:
        # [pqf, delta, lambd, alpha]
        res = []
        
        # начинаем подбор
        for d in range(delta, 0, -1):
            l = lambd
            while(l>0):
                a = (lambd-lambd/10)
                while(a>0):
                    # обучаем нейронную сеть на текущих параметрах и 
                    # получаем функционал качества разбиения
                    func = self.__functional_main(data, l, a, d, count_of_clusters)
                    
                    # записываем результат
                    res.append([func, d, l, a])
                # изменяем параметры    
                    a -= 0.05
                l -= 0.05
                # сообщаем время от начала подбора
                print("time of work in secod: %.2f"%(time.time() - self.__time))
        
        # сортируем массив результатов по возрастанию 
        # функционала качества разбиения
        res.sort(key=lambda x: x[0])
        
        # возвращаем строку с наименьшим 
        #функционалом качества разбиения
        return res[0]
        
    ## Функция обучения нейронной сети и разбиения данных на массивы по кластерам
    def __functional_main(self, data, lambd, alpha, delta, count_of_clusters):
        # обучаем нейронную сеть и получаем результат кластеризации
        kohonen = NNK_NRV(lambd=lambd, alpha=alpha, delta=delta, 
                          count_of_clusters=count_of_clusters)
        kohonen.fit(data)
        res = kohonen.define(data)
        
        # делим данные на массивы по кластерам
        clusters = self.__divisoin(count_of_clusters, res, data)
        
        # возвращаем  функционал качества разбиения
        return self.__fuctional_amount(clusters)
    
    ## Функция деления данных на массивы по кластерам
    def __divisoin(self, count_of_clusters, class_ar, data):
        # делим данные на массивы по кластерам
        clusters = []
        for i in range(count_of_clusters):
            clusters.append([])

        for i in range(len(data)):
            clusters[class_ar[i]].append(data[i])
        return clusters

    
    ## Функция подсчета функционала качества разбиения
    def __fuctional_amount(self, clusters):
        full_func = 0
        
        for num_of_cl in range(len(clusters)):
            func = 0
            cluster = clusters[num_of_cl]
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    func += self.__distance(cluster[i], cluster[j])**2
            full_func+=func
        return full_func
    
    ## Функция подсчета евклидова растояния
    def __distance(self, x, y):
            r = 0
            for i in range(len(x)):
                r = r + ((x[i] - y[i])**2)     #сумма квадратов расстояний
            r = np.sqrt(r)
            return r


# В данном классе было решено использовать в качестве функционала качества разбиения "Суммы квадратов попарных внутриклассовых расстояний между элементами".

# # Обучаем нашу нейронную сеть

# ## Подбираем оптимальные параметры

# In[12]:


# Количество кластеров
count_of_clusters = 4

agc = auto_generate_const()
agc.fit(X_train, count_of_clusters)


# ## Получаем оптимальную нейронную сеть и выводим значение функционала качества разбиения

# In[14]:


# выделим отдельно классы (по предыдущим работам) элементов
dfList = dataset['Class'].as_matrix()


# In[15]:


def my_func(data):
    data = data.copy()
    for i in range(len(data)):
        data[i] -= 1
    return data


# In[17]:


ar = np.array(normilize_data.drop(columns=["Class", "Type"]))

print ("функционал качества разбиения для Ward равен = %.3f"%agc.colculate_functional(count_of_clusters, 
                                                                                      my_func(dfList), ar))

print ("функционал качества разбиения для neuron network from statistica равен = %.3f"%agc.colculate_functional(count_of_clusters, 
                                                                                      my_func(dataset['Class'].as_matrix()), 
                                                                                                                ar))


# In[18]:


kohonen = agc.get_neuron_network()
print ("функционала качества разбиения равен %.3f"%agc.get_functional())


# ### Итак, сейчас наша сеть обучена. Наконец, мы можем сравнить результаты нашей классификации с фактическими значениями из фрейма данных:

# In[19]:


Y_train = kohonen.define(X_train)
Y_test = kohonen.define(X_test)
Y = [-1]*len(normilize_data)
for type_ in normilize_data.Type.unique():
    y = None
    if type_ == "Train":
        y = Y_train
    else:
        y = Y_test
    j = 0
    for i in normilize_data[normilize_data.Type == type_].index:
        Y[i]=y[j]
        j += 1


# In[20]:


DS = Y.copy()
i = 0
for i in range(len(DS)):
    DS[i] = [DS[i]+1, dfList[i]]
    i = i + 1

result = {'Name': [], 'Kohonen':[], 'Ward' :[]}

for i in range(len(dataset)):
    result['Name'].append(dataset['Наименование'][i])
    result['Kohonen'].append(DS[i][0])
    result['Ward'].append(DS[i][1])

pd_res = pd.DataFrame(result)
pd_res["Type"] = normilize_data["Type"]
pd_res


# In[21]:


print(len(pd_res[pd_res.Type == "Test"]))
print(len(pd_res[pd_res.Type == "Train"]))
pd_res[pd_res.Type == "Test"]


# #### Мы видим, что класс «2» полностью перекрывается с классом « Irissetosa », класс «1» в основном перекрывается с классом « Iris-versicolor », а класс «0» в основном перекрывается с « Iris-virginica ». Между классами « Iris-virginica » и « Iris-versicolor » существует 18 несоответствий , что соответствует 12% всего набора данных. Итак, общее соответствие наших результатов фактическим данным составляет 88%.
# 
# #### Поскольку мы инициализировали классы «0», «1», «2» случайным образом, порядок этих классов может быть другим на других итерациях этой программы, но это не меняет структуру классификации.

# In[22]:


pd_res.to_excel(writer, sheet_name = 'Result_Kohonen')
writer.save()

