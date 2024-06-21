import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils


train_data = dataset.MNIST(root='data/mnist',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_data = dataset.MNIST(root='data/mnist',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

print('>>> now, data is ready...')

train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)
print('>>> data loader is ready...')


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc = torch.nn.Linear(14 * 14 * 32, 10)
    
    def forward(self, x):
        out = self.conv(x)
        # batch size * c * h * w
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
    

cnn = CNN()
cnn = cnn.cuda()
print('>>> 卷积神经网络初始化完毕...')

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

print('>>> 训练开始....')
train_total = len(train_data)

for epoch in range(10):
    print(f'>>> 第{epoch+1}轮训练开始....')
    train_loss = 0
    for i, (images, lables) in enumerate(train_loader):
        images = images.cuda()
        lables = lables.cuda()

        outputs = cnn(images)

        loss = loss_func(outputs, lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f'  >>>> at {i} / {train_total // 64} loss is {loss.item()}')
        train_loss += loss.item()
        pass
    train_loss = train_loss / 64
    print(f'>>> 结束第{epoch+1}轮训练....训练损失：{train_loss}')

    print(f'>>> 第{epoch+1}轮 [评估] 开始....')
    test_loss = 0
    accuracy = 0
    for i, (images, lables) in enumerate(test_loader):
        images = images.cuda()
        lables = lables.cuda()

        outputs = cnn(images)

        test_loss += loss_func(outputs, lables)
        # 返回第1维的最大值 的索引和数值
        _, pred = outputs.max(1)
        accuracy += (pred == lables).sum().item()

    accuracy = accuracy / len(test_data)
    test_loss = test_loss / len(test_data) // 64
    print(f'>>> 第{epoch+1}轮 [评估] 结束....正确率{accuracy}, 损失{test_loss.item()}')
    pass


torch.save(cnn, 'model/mnist_model.pkl')
