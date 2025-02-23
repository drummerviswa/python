#dataset = titanicsurvival.csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
col=["Pclass","Sex","Age","Fare","Survived"]
ds = pd.read_csv("./data/titanicsurvival.csv")
print("Dimensions: \n",ds.shape)
print("Head: \n",ds.head(5))
print("Info: \n")
ds.info()
sns.countplot(x="Survived",data=ds)
plt.show()
sns.countplot(x="Survived",data=ds)
plt.show()
#sns.heatmap(ds.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")
#plt.show()
miss_val=ds.columns[ds.isna().any()]
print(miss_val)
"""Preprocessing:"""
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
#fit() - train datas are feeded to fit() for training the model's input
#Learned model
dt=dtmodel.fit(x_train,y_train)
#For testing the model with sample inputs we give x_test as input
y_pred=dt.predict(x_test)
print(y_pred)
#Checking the predicted value is matches the actual value or not
print("Accuracy is",accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))
#Confusion matrix for y_test and y_pred for predicted and actual data
cm=confusion_matrix(y_test,y_pred)
print(cm)