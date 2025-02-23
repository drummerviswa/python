#Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split

ds = pd.read_csv("./data/titanicsurvival.csv")
miss_val=ds.columns[ds.isna().any()]
print(miss_val)
ds.Age=ds.Age.fillna(ds.Age.mean())
ds['Sex']=ds['Sex'].map({'female':0, 'male':1}).astype(int)

x=ds.drop('Survived',axis='columns')
y=ds.Survived
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
adaboost=AdaBoostClassifier(n_estimators=5)
ad=adaboost.fit(x_train,y_train)

y_pred=ad.predict(x_test)
print("Accuracy is :",metrics.accuracy_score(y_test,y_pred)*100)


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv("./data/titanicsurvival.csv")
print(ds.shape)
print(ds.head(5))
ds.info()
miss_val=ds.columns[ds.isna().any()]
print(miss_val)
ds.Age=ds.Age.fillna(ds.Age.mean())
ds['Sex']=ds['Sex'].map({'female':0,'male':1}).astype(int)

x=ds.iloc[:,0:4].values
y=ds.iloc[:,4].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
rf_model=RandomForestClassifier(random_state=0)
rf=rf_model.fit(x_train,y_train)

y_pred=rf.predict(x_test)
print("\nModel Accuracy is :{0:0.4f}".format(metrics.accuracy_score(y_test,y_pred)*100))

for i in range(3):
    tree.plot_tree(rf.estimators_[i],max_depth=3,feature_names=['Pclass','Sex','Age','Fare'],filled=True)
    plt.show()