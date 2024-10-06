class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm1d(num_features=32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm1d(num_features=64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm1d(num_features=256)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 9, 500)  # 9 is from the output of the last pooling layer
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(500, 100)
        self.dropout2 = nn.Dropout(0.1)
        self.output = nn.Linear(100, 1)  # Output layer

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 32, 185)
        x = nn.ReLU()(x)
        x = self.pool(x)   # (batch_size, 32, 92)
        x = self.batch_norm1(x)

        x = self.conv2(x)  # (batch_size, 64, 90)
        x = nn.ReLU()(x)
        x = self.pool(x)   # (batch_size, 64, 45)
        x = self.batch_norm2(x)

        x = self.conv3(x)  # (batch_size, 128, 43)
        x = nn.ReLU()(x)
        x = self.pool(x)   # (batch_size, 128, 21)
        x = self.batch_norm3(x)

        x = self.conv4(x)  # (batch_size, 256, 19)
        x = nn.ReLU()(x)
        x = self.pool(x)   # (batch_size, 256, 9)
        x = self.batch_norm4(x)

        x = self.flatten(x) # Flatten to (batch_size, 256 * 9)
        x = self.fc1(x)     # Fully connected layer
        x = nn.ReLU()(x)
        x = self.dropout1(x) # Dropout
        x = self.fc2(x)     # Fully connected layer
        x = nn.ReLU()(x)
        x = self.dropout2(x) # Dropout
        x = self.output(x)  # Output layer
        
        return x
    


class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), padding='same')
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), padding='same')
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 3), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(num_features=32)
        
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), padding='same')
        
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding='same')
        
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), padding='same')

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 17 * 5, 700)  # Output size after convolutions
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(700, 283)  # Output size matches input width

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = nn.ReLU()(x)

        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.batch_norm3(x)

        x = self.conv6(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.conv7(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.conv8(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.output(x)

        return x