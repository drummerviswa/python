import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('./data/incomespent.csv')

print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))

Income=dataset['INCOME'].values
Spend=dataset['SPEND'].values
x=list(zip(Income,Spend))
#print(x)

from sklearn.cluster import KMeans
km=KMeans(n_clusters=3,random_state=0)
y_means=km.fit_predict(x)

plt.scatter(Income,Spend,c=km.labels_)
plt.show()

y=np.array(list(zip(Income,Spend)))
#print(y)
km1=KMeans(n_clusters=4,random_state=0)
y_means=km1.fit_predict(y)

plt.scatter(y[y_means==0,0],y[y_means==0,1],s=50, c='purple' , label='1')
plt.scatter(y[y_means==1,0],y[y_means==1,1],s=50, c='green' , label='2')
plt.scatter(y[y_means==2,0],y[y_means==2,1],s=50, c='blue' , label='3')
plt.scatter(y[y_means==3,0],y[y_means==3,1],s=50, c='pink' , label='4')
#plt.scatter(y[y_means==4,0],y[y_means==4,1],s=50, c='cyan' , label='5')

plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:1],s=100,marker='s',c='red',label='Centroids')
plt.title('Income Spent Analysis')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.legend()
plt.show()