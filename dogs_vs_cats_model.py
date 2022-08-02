import torch.nn as nn

class DogsVsCats(nn.Module): 
    def __init__(self, image_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding='same', bias=False)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding='same', bias=False)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        image_size //= 4

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same', bias=False)
        self.bnorm3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same', bias=False)
        self.bnorm4 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        image_size //= 2

        self.fc1 = nn.Linear(8 * image_size * image_size, 256)
        self.drop1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(256, 32)
        self.drop2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.image_size = image_size

    def forward(self, x):
        x = self.bnorm1(self.conv1(x))
        x = self.bnorm2(self.conv2(x))
        x = self.relu(self.pool1(x))
        
        x = self.bnorm3(self.conv3(x))
        x = self.bnorm4(self.conv4(x))
        x = self.relu(self.pool2(x))

        x = x.reshape(-1, 8 * self.image_size * self.image_size)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x