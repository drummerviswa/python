from sklearn_extra.cluster import KMedoids
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('./data/incomespent.csv')

print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))

Income=dataset['INCOME'].values
Spend=dataset['SPEND'].values
X=np.array(list(zip(Income,Spend)))
kmedoids=KMedoids(n_clusters=3,random_state=32)
kmedoids.fit(X)

plt.figure(figsize=(7.5,3.5))
plt.scatter(X[:,0],X[:,1],c=kmedoids.labels_)
plt.scatter(kmedoids.cluster_centers_[:,0],kmedoids.cluster_centers_[:,1],marker="x",color="red")
plt.show()