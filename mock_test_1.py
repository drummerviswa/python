import pandas as pd
import matplotlib.pyplot as plt
data_csv = pd.read_csv("./data/marks.csv",index_col=0)
print(data_csv,"\n")

print("Shape of the file: ",data_csv.shape)
print("First 3 tuples: \n",data_csv.head(3))
print("Last 3 tuples: \n",data_csv.tail(3))

physics = data_csv["Physics"]
chemistry = data_csv["Chemistry"]
maths = data_csv["Maths"]

#Statistical Description
print(physics.describe())
print(chemistry.describe())
print(maths.describe())

#Histogram
plt.hist(physics)
plt.title("Physics Histogram")
plt.xlabel("Marks")
plt.show()

#Scatterplot for Physics and Chemistry
plt.scatter(physics,chemistry)
plt.title("Scatterplot for Physics and Chemistry")
plt.xlabel("Physics")
plt.ylabel("Chemistry")
plt.show()

#Scatterplot for Chemistry and Maths
plt.scatter(chemistry,maths)
plt.title("Scatterplot for Chemistry and Maths")
plt.xlabel("Chemistry")
plt.ylabel("Maths")
plt.show()

#Scatterplot for Physics and Maths
plt.scatter(physics,maths)
plt.title("Scatterplot for Physics and Maths")
plt.xlabel("Physics")
plt.ylabel("Maths")
plt.show()

#Boxplot for Physics
plt.boxplot(physics)
plt.title("Boxplot for Physics")
plt.show()

#Boxplot for Chemistrt
plt.boxplot(chemistry)
plt.title("Boxplot for Chemistry")
plt.show()

#Boxplot for Maths
plt.boxplot(maths)
plt.title("Boxplot for Maths")
plt.show()

data_csv.plot.box()
plt.title("Box Plot for all")
plt.show()