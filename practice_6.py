"""
    - cross_val_score
    - cross_val_predict
    - Kfold
    - LeaveOneOut
"""

#dataset = titanicsurvival.csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
col=["Pclass","Sex","Age","Fare","Survived"]
ds = pd.read_csv("./data/titanicsurvival.csv")
# print("Dimensions: \n",ds.shape)
# print("Head: \n",ds.head(5))
# print("Info: \n")
# ds.info()
# sns.countplot(x="Survived",data=ds)
# plt.show()
# sns.countplot(x="Survived",data=ds)
# plt.show()
#sns.heatmap(ds.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")
#plt.show()
"""Preprocessing:"""
miss_val=ds.columns[ds.isna().any()]
print(miss_val)
#Replacing missing values of age with mean of the age
ds.Age=ds.Age.fillna(ds.Age.mean())
#Converting Sex col's attribute into Binary attribute
ds["Sex"]=ds["Sex"].map({'female':0,'male':1}).astype(int)
"""Splited it into two data set until Survived col
    Parts = 4
            I-x                 II-y
    Pclass,Sex,Age,Fare       Survived
"""
#sns.countplot(ds["Survived"])
x=ds.drop("Survived",axis="columns")
y=ds.Survived
print(x)
print(y)
#Test size = 20% remaining 80% is for Train data, random state = 0 the output will be same whenever it runs
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#Classification type is choosed as Decision tree
dtmodel=DecisionTreeClassifier()

"""Model Selection"""
#cv=5 //default
#cross_val_score
from sklearn.model_selection import cross_val_score
val_score = cross_val_score(dtmodel, x, y)
print("Y predict of cross_val_predict",val_score)
print("Cross Val Score Mean: ",np.mean(val_score))

#cross_val_predict
from sklearn.model_selection import cross_val_predict
val_pred = cross_val_predict(dtmodel, x, y)
print("Y predict of cross_val_predict",val_pred)
print("Cross Val Score Mean: ",np.mean(val_pred))

#Kfold
from sklearn.model_selection import KFold
no_of_fold = int(input("Enter no.of folds: "))
kf = KFold(n_splits=no_of_fold)
kf.get_n_splits(x)
kf_accuracy = 0
for i, (train_index, test_index) in enumerate(kf.split(x)):
    trainX,testX = x.take(list(train_index),axis=0),x.take(list(test_index),axis=0)
    trainY,testY = y.take(list(train_index),axis=0),y.take(list(test_index),axis=0)
    #Model fitting
    dt=dtmodel.fit(trainX,trainY)
    #Predicting Y with testX
    y_predict = dt.predict(testX)
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    #Calculating accuracy by accuracy_score with testY and predicted Y
    acc=round(accuracy_score(testY,y_predict)*100,2)
    #Summing up the accuracy in a variable
    kf_accuracy+=acc
#Dividing the summed accuracy with no.of fold
print(f"Accuracy: ",round(kf_accuracy/no_of_fold,2))

Pclass = int(input("Enter Person's PClass number: "))
Sex = int(input("Enter Person's Gender:(0-Female,1-Male) "))
Age = int(input("Enter Person's Age: "))
Fare = float(input("Enter Person's Fare: "))
Person = [[Pclass,Sex,Age,Fare]]
Result = dt.predict(Person)
print(Result)

#LeaveOneOut
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(x)
loo_accuracy = 0
for i, (train_index, test_index) in enumerate(loo.split(x)):
    trainX,testX = x.take(list(train_index),axis=0),x.take(list(test_index),axis=0)
    trainY,testY = y.take(list(train_index),axis=0),y.take(list(test_index),axis=0)
    #Model fitting
    dt=dtmodel.fit(trainX,trainY)
    #Predicting Y with testX
    y_predict = dt.predict(testX)
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    #Calculating accuracy by accuracy_score with testY and predicted Y
    acc=round(accuracy_score(testY,y_predict)*100,2)
    #Summing up the accuracy in a variable
    loo_accuracy+=acc
#Dividing the summed accuracy with no.of fold
print(f"Accuracy: ",round(loo_accuracy/no_of_fold,2))

Pclass = int(input("Enter Person's PClass number: "))
Sex = int(input("Enter Person's Gender:(0-Female,1-Male) "))
Age = int(input("Enter Person's Age: "))
Fare = float(input("Enter Person's Fare: "))
Person = [[Pclass,Sex,Age,Fare]]
Result = dt.predict(Person)
print(Result)

if Result == 1:
    print("Person might be survived")
else:
    print("Person might not be survived")