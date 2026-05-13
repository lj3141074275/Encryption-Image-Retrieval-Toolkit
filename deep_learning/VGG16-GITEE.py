import torch
from torch import nn,optim,tensor
from torch.utils.data import DataLoader
from torchvision.utils import  make_grid
from torchvision import datasets,transforms
import numpy as np
from matplotlib import pyplot as plt
import time

#全局变量
batch_size=32   #每次喂入的数据量
# num_print=int(50000//batch_size//4)
num_print=100
epoch_num=30 #总迭代次数
lr=0.01
step_size=10  #每n次epoch更新一次学习率
#数据获取(数据增强,归一化)
def transforms_RandomHorizontalFlip():
    transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.485,0.456,0.406),(0.226,0.224,0.225))])#广泛均值和方差
    train_dataset=datasets.CIFAR10(root='../../data_hub/plainimage',train=True,transform=transform_train,download=True)
    test_dataset=datasets.CIFAR10(root='../../data_hub/plainimage',train=False,transform=transform,download=True)
    return train_dataset,test_dataset

#数据增强:随机翻转
train_dataset,test_dataset=transforms_RandomHorizontalFlip()

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#模型,优化器
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from ch5_VGG16 import *
model=Vgg16_net().to(device)

#在多分类情况下一般使用交叉熵
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.8,weight_decay=0.001)
schedule=optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)

#训练
loss_list=[]  #为了后续画出损失图
start=time.time()

#train
for epoch in range(epoch_num):
    ww = 0
    running_loss=0.0
    #0是对i的给值(循环次数从0开始技术还是从1开始计数的问题):
    #???
    for i,(inputs,labels) in enumerate(train_loader,0):
        #将数据从train_loader中读出来,一次读取的样本是32个
        inputs,labels=inputs.to(device),labels.to(device)
        #用于梯度清零,在每次应用新的梯度时,要把原来的梯度清零,否则梯度会累加
        optimizer.zero_grad()
        outputs = model(inputs)
        loss=criterion(outputs,labels).to(device)
        #反向传播,pytorch会自动计算反向传播的值
        loss.backward()
        #对反向传播以后对目标函数进行优化
        optimizer.step()
        running_loss+=loss.item()
        loss_list.append(loss.item())
        if(i+1)%num_print==0:
            print('[%d epoch,%d]  loss:%.6f' %(epoch+1,i+1,running_loss/num_print))
            running_loss=0.0
    lr_1=optimizer.param_groups[0]['lr']
    print("learn_rate:%.15f"%lr_1)
    schedule.step()
end=time.time()
print("time:{}".format(end-start))

#测试
#由于训练集不需要梯度更新,于是进入测试模式
model.eval()
correct=0.0
total=0
with torch.no_grad(): #训练集不需要反向传播
    print("=======================test=======================")
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)
        print(outputs)
        pred=outputs.argmax(dim=1)  #返回每一行中最大值元素索引
        total+=inputs.size(0)
        correct+=torch.eq(pred,labels).sum().item()
print("Accuracy of the network on the 10000 test images:%.2f %%" %(100*correct/total) )
print("===============================================")

