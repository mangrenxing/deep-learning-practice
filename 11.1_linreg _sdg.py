import torch, random
torch.manual_seed(42)
random.seed(42)

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
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

# 1. 生成数据
true_w = torch.tensor([2, -3.4]) 
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 2. 初始化模型参数（3.2.3）
w = torch.ones((2,1), requires_grad=True)
print(w)
b = torch.zeros(1, requires_grad=True)
lr=0.03  #学习步长
num_epochs=3  
batch_size=10  #用于把累加的梯度初一batch大小

step=0
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=squared_loss(linreg(X,w,b),y)
        l.sum().backward()
        if step%20==0:
           print("w.grad:",w.grad,"b.grad:",b.grad,"step:",step)
        step+=1
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l=squared_loss(linreg(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')

print(w)

