import torch
import random 

# 生成数据集
def synthetic_data(w,b,num_examles):
    x=torch.normal(0,1,(num_examles,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape(-1,1)

# 定义数据迭代器
def data_iter(batch_size,features,labels):
    num_examles=len(features)
    indices=list(range(num_examles))
    random.shuffle(indices)
    for i in range(0,num_examles,10):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examles)]).long()
        yield features[batch_indices],labels[batch_indices]

# 定义线性回归模型
def linreg(x,w,b):
    return torch.matmul(x,w)+b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#定义优化算法
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            params-=lr*param.grad/batch_size
            param.grad.zero_()

# 1. 生成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 2. 初始化模型参数（3.2.3）
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 3. 尝试跑一个批次，看Loss能不能算出来
for X, y in data_iter(10, features, labels):
    y_hat = linreg(X, w, b)          # 调用模型
    loss = squared_loss(y_hat, y)    # 调用损失函数
    print("当前批次的预测值 y_hat:", y_hat)
    print("当前批次的损失 loss:", loss)
    break  # 只看第一批，防止刷屏



