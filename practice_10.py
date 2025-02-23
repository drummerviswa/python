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

model=KMeans(n_clusters=2,random_state=0)
y_means=model.fit_predict(X)

plt.scatter(Income,Spend,c=model.labels_)
plt.show()

plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='purple' , label='1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='green' , label='2')

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:1],s=100,marker='s',c='red',label='Centroids')
plt.title('Income Spent Analysis')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.legend()
plt.show()