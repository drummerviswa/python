"""
    
"""
import pandas as pd
data = pd.read_csv("./data/weather.csv")
print(data.shape)

# Method 1 using map function
ds1=data.copy()
ds1.info()
ds1["play"]=ds1["play"].map({"yes":1,"no":0}).astype(str).astype(int)
print(ds1.play)
ds1["humidity"]=ds1["humidity"].map({"high":1,"normal":0}).astype(str).astype(int)
print(ds1.humidity)
ds1["outlook"]=ds1["outlook"].map({"sunny":0,"overcast":1,"rainy":2}).astype(str).astype(int)
print(ds1.outlook)
ds1["temperature"]=ds1["temperature"].map({"hot":0,"mild":1,"cool":2}).astype(str).astype(int)
print(ds1.temperature)
ds1["windy"]=ds1["windy"].map({"TRUE":1,"FALSE":0}).astype(bool).astype(int)
print(ds1.windy)
ds1.info()

# Method 2 get dummy function
ds2=data.copy()
ds2.info()
ds2.shape
ds2 = pd.get_dummies(ds2,columns=["temperature","humidity","windy","outlook"],dtype=int)
print(ds2.shape)
print(ds2.head(5))
# x=ds1.drop("play",axis="column")
ds2["play"]=ds2["play"].map({"yes":1,"no":0})
y=ds2["play"]
print(y)
ds2.info()

# Method 3 using LabelEncoder
ds3=data.copy()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#ds3=ds3.drop(columns=["play"])
for x in ds3.columns:
    ds3[x]=le.fit_transform(ds3[x])
print(ds3)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x="play",data=ds1)
plt.show()
print(ds1["play"].value_counts())
print(pd.crosstab(ds1["play"],columns="count",normalize=True))

# Decision Tree Classifier
# criterion["gini","entropy","log_loss"] default "gini"
from sklearn.tree import DecisionTreeClassifier
x = ds1.drop("play",axis="columns")
y=ds1.play
# print(x)
# print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
dtmodel=DecisionTreeClassifier(criterion="gini")
dt=dtmodel.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
print("Decision Tree Accuracy: ",accuracy_score(y_test,y_pred))

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nbmodel=GaussianNB()
x=ds3.drop("play",axis="columns")
y=ds3.play
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
nb=nbmodel.fit(x_train,y_train)
y_pred=nb.predict(x_test)
print(y_pred)