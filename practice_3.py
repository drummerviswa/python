import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(170,10,250)

#print(x)

#plt.hist(x)
#plt.show()

# data = np.array([10,30,30,40,40,45,45,45,45,36,23,89,54,23,1,23,23,76])
# plt.hist(data)
# plt.show()

# x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
# y=[99,86,87,88,111,86,103,87,94,78,77,85,86]

# plt.scatter(x,y)
# plt.show()

x=np.random.normal(5.0,1.0,1000)
y=np.random.normal(10.0,2.0,1000)
print(x)
print(y)

plt.scatter(x,y)
plt.show()

#describe - numerical
#describe(include="o") - categorical