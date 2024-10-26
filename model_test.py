import torch
from torch import nn

from data_loader import data_loader


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=filter_siz1,kernel_size=5,padding=2)
        # 第一层卷积层,输入是维度为3的RGB图片,输出是维度为32的向量
        # (32x32x3→32x32x32)
        # (nxn变为(n-f+2p+s)x(n-f+2p+s),f=5,p=2,s默认=1,故前后矩阵尺寸不变)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)
        # 池化,(32x32x32→16x16x32)((n-f)/s+1,相当于尺寸减半)
        self.bn1 = nn.BatchNorm2d(filter_siz1)
        # 正则化
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=filter_siz1,out_channels=filter_siz1,kernel_size=5,padding=2)
        # 第二层卷积层,输入是维度为32的向量,输出是维度为32的向量
        # (16x16x32→16x16x32)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        # 池化,(16x16x32→8x8x32)
        self.bn2 = nn.BatchNorm2d(filter_siz1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=filter_siz1,out_channels=filter_siz2,kernel_size=5,padding=2)
        # 第三层卷积层,输入是维度为32的向量,输出是维度为64的向量
        # (8x8x32→8x8x64)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2)
        # 池化,(8x8x32→4x4x64)
        self.bn3 = nn.BatchNorm2d(filter_siz2)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        # 全部压成一维扁平化，便于全连接层处理
        self.linear1 = nn.Linear(filter_siz2 * 4 * 4,filter_siz2)
        self.dropout = nn.Dropout(dropout_rate)
        # dropout,防止过拟合
        self.linear2 = nn.Linear(filter_siz2,num_outputs)


    def forward(self,x):
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.maxPool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.maxPool3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 定义数据集的类别
classes = ('automobile','airplane','bird','cat',
           'deer','dog','frog','horse','ship','truck','nailong')
img_siz,num_outputs = 32,11
filter_siz1 = 32
filter_siz2 = 64
dropout_rate = 0.25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size,learning_rate,num_epochs = 20,0.0001,20
output_step = 500
print(device)
model = Net().to(device)
train_org_loader,test_org_loader,train_nailong_loader,test_nailong_loader = data_loader(batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

model.load_state_dict(torch.load('checkpoint/model.pth'))

model.eval()
# 将模型设置为评估模式
test_loss = 0.0
correct = 0.0
total = 0
with torch.no_grad():
    for i,data in enumerate(test_org_loader):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        test_loss += loss.item()
        total += labels.size(0)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).sum().item()
print("correct=%d,total=%d,acc=%.4f" % (correct,total,correct * 1.0 / total))
test_loss = 0.0
correct = 0.0
total = 0
with torch.no_grad():
    for i,data in enumerate(test_nailong_loader):
        images,labels = data
        labels = torch.full((labels.size()),10)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        test_loss += loss.item()
        total += labels.size(0)
        _,predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).sum().item()
print("correct=%d,total=%d,acc=%.4f" % (correct,total,correct * 1.0 / total))
