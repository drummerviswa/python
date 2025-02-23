from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import pandas as pd
from sklearn.model_selection import train_test_split

ds = pd.read_csv("./data/titanicsurvival.csv")
miss_val=ds.columns[ds.isna().any()]
print(miss_val)
ds.Age=ds.Age.fillna(ds.Age.mean())
ds["Sex"]=ds["Sex"].map({'female':0,'male':1}).astype(int)

x=ds.drop("Survived",axis="columns")
y=ds.Survived

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
estimator=[]
dtmodel=DecisionTreeClassifier(criterion="entropy",max_depth=3)
estimator.append(("Decision Tree",dtmodel))

log_reg = LogisticRegression(solver="liblinear")
estimator.append(("Logistic",log_reg))

svm_clf = SVC(gamma="scale")
estimator.append(("SVM",svm_clf))

print(estimator)
voting=VotingClassifier(estimators=estimator)
vt=voting.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = vt.predict(x_test)
print("Accuracy is",accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
cm_dis = ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()