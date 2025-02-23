import pandas as pd

data_csv = pd.read_csv("./data/Toyota.csv")

#Two types of copying a dataset
#Shadow copy - single memory location => .copy(deep=false)
shadow_copy = data_csv.copy(deep=False)
#Deep Copy -  another memory location => .copy(deep=true)
deep_copy = data_csv.copy(deep=True)

