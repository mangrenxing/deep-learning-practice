import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
def synthetic_data(w,b,num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))
    #torch.normal是从正态分布中随机采样，0是均值，1是标准差，(num_examples,len(w))是张量的形状
    y=torch.matmul(x,w)+b  
    #torch.matmul是矩阵乘法,把x乘以w的权重向量
    y+=torch.normal(0,0.01,y.shape)  
    #+=是原地修改，给y加上均值为0，标准差为0.01的随机误差，y.shape表示会自动广播成y的形状，保证y的每一个样本加上的是完全独立的随机数。如果不写则会导致广播会把这个单一数字复制1000次。
    return x,y.reshape(-1,1)
    #强制把y变成列向量，-1表示行数自动推断，1表示列数为1。如果没有y.reshape(-1,1)，当代码进行损失计算时，pytorch会触发广播机制， (1000,) 和 (1000, 1) 强行拉伸对齐，最后得到一个 (1000, 1000) 的巨型矩阵。整个计算直接崩溃。

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
#synthetic_data函数生成1000个样本，特征维度为2，标签为1维。features是一个1000x2的张量，labels是一个1000x1的张量。

print('features:',features[0],'\nlabel:',labels[0])

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    #把1000个数字0-999放进一个列表中。

    random.shuffle(indices)
    #随机打乱列表中的数字，保证每次迭代返回的样本顺序不同。
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)]).long()
        #min(i+batch_size,num_examples)保证最后一个batch的样本数可能小于batch_size。
        print("当前批次的索引:", batch_indices)
        print("提取后的特征形状:", features[batch_indices].shape)
        print("提取后的标签形状:", labels[batch_indices].shape)        
        yield features[batch_indices],labels[batch_indices]
        #yield是一个生成器函数，返回一个迭代器，每次调用next()时返回一个batch的特征和标签。
        #features[batch_indices]是一个(batch_size,2)的张量，labels[batch_indices]是一个(batch_size,1)的张量。
        
batch_size=10  #batch_size是每次迭代返回的样本数

for X,y in data_iter(batch_size,features,labels):
    print(X,y)
    break