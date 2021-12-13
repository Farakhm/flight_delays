from math import e
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle

warnings.filterwarnings('ignore')

train_df=pd.read_csv("flight_delays_train_b.csv")
print(train_df.head())

# Делаем данные числовыми
# Меняем N на 1, а Y - на 0
train_df.loc[train_df['dep_delayed_15min'] == 'N', 'dep_delayed_15min'] = 0
train_df.loc[train_df['dep_delayed_15min'] == 'Y', 'dep_delayed_15min'] = 1
# Меняем месяцы и дни на числа
dataset = train_df
dataset['Month'] = dataset['Month'].replace(['c-1','c-2','c-3','c-4',
                'c-5','c-6','c-7','c-8','c-9','c-10','c-11','c-12'], [1,2,3,4,5,6,7,8,9,10,11,12])
dataset['DayofMonth'] = dataset['DayofMonth'].replace(['c-1','c-2','c-3','c-4','c-5','c-6','c-7',
                'c-8','c-9','c-10','c-11','c-12','c-13','c-14','c-15','c-16','c-17','c-18','c-19',
                'c-20','c-21','c-22','c-23','c-24','c-25','c-26','c-27','c-28','c-29','c-30','c-31'],
                [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
dataset['DayOfWeek'] = dataset['DayOfWeek'].replace(['c-1','c-2','c-3','c-4',
                'c-5','c-6','c-7'], [1,2,3,4,5,6,7])
# Вводим два новых столбца - часы и минуты отправления
dataset['Hours'] = dataset['DepTime']//100
dataset['Minutes'] = dataset['DepTime']%100
dataset['Hours'] = dataset['Hours'].replace([21,22,23,0,1,2,3,4,5,6,7,8], 1)
dataset['Hours'] = dataset['Hours'].replace([9,10,11,12,13,14,15,16,17,18,19,20], 2)

# Делаем списки перевозчиков, аэропортов назначения и прибытия
def lsttodict(lst):
    lst = set(lst)
    lst = list(lst)
    lst.sort()
    res1 = []
    res2 = []
    for i in range(len(lst)):
        res1.append(lst[i])
        res2.append(i)
    return res1, res2

def num_makes(lst):
    comp = list(lst)
    omp = set(comp)
    omp = list(omp)
    omp.sort()
    num_comp = len(omp)
    res = []
    for i in range(num_comp):
        r = comp.count(omp[i])
        res.append(r)
    return res


compname, compnum = lsttodict(dataset['UniqueCarrier'])
depname, depnum = lsttodict(dataset['Origin'])
arrname, arrnum = lsttodict(dataset['Dest'])

compflights = num_makes(dataset['UniqueCarrier'])
depflights = num_makes(dataset['Origin'])
arrflights = num_makes(dataset['Dest'])
dataset['Carrier'] = dataset['UniqueCarrier'].replace(compname, compflights)
dataset['Depart'] = dataset['Origin'].replace(depname, depflights)
dataset['Arriv'] = dataset['Dest'].replace(arrname, arrflights)

# Заменяем буквенные обозначения числовыми из словарей

dataset['UniqueCarrier'] = dataset['UniqueCarrier'].replace(compname, compnum)
dataset['Origin'] = dataset['Origin'].replace(depname, depnum)
dataset['Dest'] = dataset['Dest'].replace(arrname, arrnum)

# Разбиваем на категории
dataset['Month'] = pd.cut(dataset['Month'], bins=[0,1,2,3,4,5,6,7,8,9,10,11,12],
                                            labels=['Ja','Fe', 'Mr', 'Ap','Ma', 'Jn', 'Jl', 'Au', 'Se', 'Oc','No','De'])
dataset['Dist_bin'] = pd.cut(dataset['Distance'], bins=[0.,750.,1500., 5000.],
                                                    labels=[ 'Short', 'Middle', 'Long'])
# [0.,750.,1500., 5000.],  0.5455                                                    
dataset['Carrier_Bin'] = pd.cut(dataset['Carrier'], bins=[0,1500,4500,20000],
                                                    labels=['Few', 'Little', 'Large'])
dataset['Depart_Bin'] = pd.cut(dataset['Depart'], bins=[0,100,600,1000,6000],
                                                    labels=['Few', 'Little', 'Normal','Large'])

dataset['Arriv_Bin'] = pd.cut(dataset['Arriv'], bins=[0,100,600,1000,6000],
                                                    labels=['Few', 'Little', 'Normal', 'Large'])


traindf=train_df
drop_column = ['Minutes', 'Origin','Dest', 'Carrier', 'UniqueCarrier', 'Origin','Dest','Depart','Arriv']
train_df.drop(drop_column, axis=1, inplace=True)

traindf = pd.get_dummies(traindf, columns=['Month','Dist_bin', 'Carrier_Bin', 'Depart_Bin', 'Arriv_Bin'],
                                   prefix=['Month','Dist', 'Carrier', 'Depart', 'Arriv'])

print(traindf.head())
# Теперь наш датасет готов к обработке
#sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
#fig=plt.gcf()
#fig.set_size_inches(20,12)
#plt.show()

all_features = traindf.drop("dep_delayed_15min",axis=1)
Targeted_feature = traindf["dep_delayed_15min"]
Targeted_feature = Targeted_feature.astype('int')
X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.4,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = RandomForestClassifier(criterion='gini', n_estimators=700,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)

model.fit(X_train,y_train)

prediction_rm=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Random Forest Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
print(roc_auc_score(y_test, prediction_rm),'*****////*****')
