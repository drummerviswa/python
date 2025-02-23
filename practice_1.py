import pandas as pd
### pd is reference for pandas lib
# data frame - file as a table
# pd.read_csv - csv file
# pd.read_excel - xlsx file
# pd.read_table - entire file

#READ CSV FILE
#data_csv = pd.read_csv("./data/iris.csv")

#READ EXCEL FILE
#data_csv = pd.read_excel("./data/iris.xlsx",sheet_name="")

#READ TEXT FILE
#data_csv = pd.read_table("./data/iris.txt")

# => First column as index <=> index_col=0
#data_csv = pd.read_csv("./data/iris.csv",index_col=0)

#Any obs have value as NAN - Not a number, it will be returned as Null value
#Any special charecters must be replaced as NAN
# ?, ###, - anything..... => na_values=["charecter1","charecter2"]
#data_csv = pd.read_csv("./data/iris.csv",index_col=0,na_values=["?","###"])

#Seperator and Delimiter
#data_csv = pd.read_csv("./data/iris.csv",sep=" ")
data_csv = pd.read_csv("./data/iris.csv",delimiter=" ")
print(data_csv)