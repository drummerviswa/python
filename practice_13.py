import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('./data/incomespent.csv')

print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))

from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=7,affinity="euclidean",linkage="average")
y_means=model.fit_predict(dataset)
print(y_means)

X=dataset.iloc[:,[0,1]].values
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='purple' , label='1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='orange' , label='2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50, c='red' , label='3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50, c='green' , label='4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=50, c='blue' , label='5')
plt.title('Income Spent Analysis')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.legend()
plt.show()

import scipy.cluster.hierarchy as clus

plt.figure(1,figsize=(16,8))
dendogram=clus.dendrogram(clus.linkage(dataset,method="single"))

plt.title("Dendogram Tree Graph")
plt.xlabel("Income")
plt.ylabel("Spent")
plt.show()