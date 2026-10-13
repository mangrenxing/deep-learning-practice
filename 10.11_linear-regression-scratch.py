import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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

print('features:',features[0],'\nlabel:',labels[0])

#绘图
plt.figure(figsize=(3.5,2.5))
plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy().squeeze(),1)
plt.show()
plt.savefig("scatter_plot.png")  # 把图保存到当前文件夹
print("图片已保存！")