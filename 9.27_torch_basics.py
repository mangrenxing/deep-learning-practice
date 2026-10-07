import torch

#创建张量
a=torch.tensor([1,2,3,4,5,6])   #tensor是为张量中的每个元素赋予确定值
b=torch.arange(0,10,2)          #arange是创建一个一维张量，包含从0，10（不包括10），其步长为2
c=torch.linspace(0,6,4)         #linspace是创建一个一维张量，包含从0到6（包括6），元素个数为4
d=torch.zeros([2,5])            #d是一个二维张量2行5列，zeros使其所有元素为0。
f=torch.ones([3,4])             #f是一个二维张量3行4列，ones使其所有元素为1。
print(a,b,c,d,f)

#张量形状
j=torch.arange(0,12)
print(j,j.shape)                #shape来访问张量的形状
h=torch.reshape(j,[3,4])        #reshape改变形状
print(h,h.shape)

#访问张量元素
print("取一个元素h[1]:",h[1,2])    #取一个元素
print("取一行h[1]:",h[1:])         #取一行元素
print("取一列h[:,1]:",h[:,1])      #取一列元素
print("取第二行第三列h[1,2]:",h[1,2])
print("子区域h[1:3,1:2]:",h[1:3,1:2])  #从第二行到第三行，第二列到第三列元素
print("子区域h[::3,::2]:",h[::3,::2])  #从第一行到最后一行，每三行一跳，每2列一跳

#广播机制
j2=torch.reshape(j,[1,4])
print("广播：j2+h:",j2+h)  #广播机制，j是1维张量，h是2维张量，j会自动扩展为与h相同的形状

#内存分配测试
x=torch.tensor([1,2,3,4,5])
y=torch.tensor([4,5,6,7,8])
before=id(y)
y=y+x
id(y)==before

y2=torch.tensor([4,5,6,7,8])
before2=id(y2)
x+=y2
id(y2)==before2

#将Numpy转化为张量
A=x.numpy()  
B=torch.tensor(A)
type(A),type(B)

#将张量转化为Numpy
C=B.numpy()
type(C)

#将大小为1的装量转换成python的张量
m=torch.tensor([3.5])
print(m,m.item(),float(m),int(m))

