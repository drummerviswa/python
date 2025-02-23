import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

ds=pd.read_csv("./data/titanicsurvival.csv")
print(ds)
print(ds.shape)
print(ds.head(5))
print(ds.info())
miss_val = ds.columns[ds.isna().any()]
print(miss_val)
ds.Age=ds.Age.fillna(ds.Age.mean())
ds["Sex"]=ds["Sex"].map({"female":0,"male":1}).astype(int)
# sns.countplot(x="Survived",data=ds)
# plt.show()
x=ds.drop("Survived",axis="columns")
y=ds.Survived
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
dtmodel=DecisionTreeClassifier()
dt=dtmodel.fit(x_train,y_train)
y_pred=dt.predict(x_test)
# print(y_pred)

# print("Ac s",accuracy_score(y_test,y_pred)*100)
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))

# from sklearn.naive_bayes import GaussianNB
# nbmodel=GaussianNB()
# nb=nbmodel.fit(x_train,y_train)
# y_pred=nb.predict(x_test)
# print(y_pred)

# from sklearn.model_selection import cross_val_predict,cross_val_score
# val_pred=cross_val_predict(dtmodel,x,y)
# val_score=cross_val_score(dtmodel,x,y)
# print(val_pred,val_score)

# from sklearn.model_selection import KFold
# no_of_fold=int(input("Enter no.of folds: "))
# kf=KFold(n_splits=no_of_fold)
# kf.get_n_splits(x)
# kf_accuracy=0
# for i,(train_index,test_index) in enumerate(kf.split(x)):
#     trainX,testX=x.take(list(train_index),axis=0),x.take(list(test_index),axis=0)
#     trainY,testY=y.take(list(train_index),axis=0),y.take(list(test_index),axis=0)
#     dt=dtmodel.fit(trainX,trainY)
#     y_pred=dt.predict(testX)
#     acc=round(accuracy_score(testY,y_pred)*100,2)
#     kf_accuracy+=acc
# print(f"Accuracy: ",round(kf_accuracy/no_of_fold,2))
# Pclass = int(input("Enter Person's PClass number: "))
# Sex = int(input("Enter Person's Gender:(0-Female,1-Male) "))
# Age = int(input("Enter Person's Age: "))
# Fare = float(input("Enter Person's Fare: "))
# Person = [[Pclass,Sex,Age,Fare]]
# print(Person)
# Result = dt.predict(Person)
# print(Result)
# if Result == 1:
#     print("Person might be survived")
# else:
#     print("Person might not be survived")

from sklearn.model_selection import LeaveOneOut
loo=LeaveOneOut()
loo.get_n_splits(x)
loo_accuracy=0
for i,(train_index,test_index) in enumerate(loo.split(x)):
    trainX,testX=x.take(list(train_index),axis=0),x.take(list(test_index),axis=0)
    trainY,testY=y.take(list(train_index),axis=0),y.take(list(test_index),axis=0)
    dt=dtmodel.fit(trainX,trainY)
    y_pred=dt.predict(testX)
    acc=round(accuracy_score(testY,y_pred)*100,2)
    loo_accuracy+=acc
print(f"Accuracy: ",round(loo_accuracy,2))
Pclass = int(input("Enter Person's PClass number: "))
Sex = int(input("Enter Person's Gender:(0-Female,1-Male) "))
Age = int(input("Enter Person's Age: "))
Fare = float(input("Enter Person's Fare: "))
Person = [[Pclass,Sex,Age,Fare]]
print(Person)
Result = dt.predict(Person)
print(Result)
if Result == 1:
    print("Person might be survived")
else:
    print("Person might not be survived")