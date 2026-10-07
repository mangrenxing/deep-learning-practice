import os
import pandas as pd
import torch

os.makedirs(os.path.join('10.4_data'),exist_ok=True)
data_file=os.path.join('10.4_data','data.csv')
with open(data_file,'w') as f:
    f.write('id,name,age,grade\n')
    f.write('1,NaN,30,NaN\n')
    f.write('2,Bob,NaN,85\n')
    f.write('3,Charlie,35,NaN\n')
    f.write('4,NaN,28,NaN\n')
    f.write('5,Eve,NaN,95\n')
    f.write('6,NaN,27,NaN\n')

data=pd.read_csv(data_file)
print(data)

most_missing=data.isnull().sum().idxmax()
data=data.drop(columns=[most_missing])
print(data)

inputs,outputs=data.iloc[:,1],data.iloc[:,2]
inputs = pd.get_dummies(inputs, dummy_na=True)
outputs=outputs.fillna(outputs.mean())
print(inputs,outputs)

x,y=torch.tensor(inputs.values,dtype=torch.float32),torch.tensor(outputs.values,dtype=torch.float32)
print(x,y)