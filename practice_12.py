from sklearn.metrics import silhouette_score
from pyclustertend import hopkins
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('./data/incomespent.csv')

print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))

print(hopkins(dataset,50))

Income=dataset['INCOME'].values
Spend=dataset['SPEND'].values
X=list(zip(Income,Spend))
from sklearn.cluster import KMeans
wcv=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    wcv.append(km.inertia_)
plt.plot(range(1,11),wcv,color="red",marker="8")
plt.title("Optimal K value")
plt.xlabel("No. of Clusters")
plt.ylabel("WCV")
plt.show()

model=KMeans(n_clusters=3,random_state=0)
y_means=model.fit_predict(X)

plt.scatter(Income,Spend,c=model.labels_)
plt.show()

ss = silhouette_score(X,model.labels_)
print(ss)