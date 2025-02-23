import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ds=pd.read_csv("./data/titanicsurvival.csv")
ds.Age=ds.Age.fillna(ds.Age.mean())
ds["Sex"]=ds["Sex"].map({"female":0,"male":1}).astype(int)

x=ds.drop("Survived",axis="columns")
y=ds.Survived

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
adaboost=AdaBoostClassifier()
ad=adaboost.fit(x_train,y_train)
y_pred=ad.predict(x_test)
print(y_pred)