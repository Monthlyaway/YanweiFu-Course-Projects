import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    """
    Basic CNN with convolutional layers, pooling, and fully-connected layers.
    """

    def __init__(self):
        # The input shape to fc1 is 128 * 4 * 4 because:
        # - 128 is the number of output channels from conv3
        # - 4 * 4 is the spatial dimensions after 3 pooling layers:
        #   - Input: 32x32 -> pool1: 16x16 -> pool2: 8x8 -> pool3: 4x4
        # Note: MaxPool2d has no trainable parameters, so reusing the same pooling layer
        # is memory efficient and doesn't affect model performance
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN with batch normalization and dropout.
    """

    def __init__(self):
        # BatchNorm2d(64) normalizes the activations of the 64 output channels from conv1
        # It doesn't change the shape of the input, only normalizes each channel
        # Shape remains (batch_size, 64, height, width)
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout(F.relu(self.bn7(self.fc1(x))))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The shortcut is defined as an empty Sequential module by default
        # This is necessary because we need a consistent interface for both cases:
        # 1. When stride=1 and in_channels=out_channels: we can directly add x
        # 2. When stride≠1 or in_channels≠out_channels: we need a projection layer
        # Using Sequential allows us to handle both cases uniformly in the forward pass
        # with out += self.shortcut(x)
        self.shortcut = nn.Sequential()
        # When stride != 1 or in_channels != out_channels, we need a projection shortcut
        # Example: if input shape is (batch, 64, 32, 32) and stride=2, out_channels=128:
        # 1. conv1: (batch, 64, 32, 32) -> (batch, 128, 16, 16) [stride=2 reduces spatial dims]
        # 2. conv2: (batch, 128, 16, 16) -> (batch, 128, 16, 16)
        # 3. shortcut: (batch, 64, 32, 32) -> (batch, 128, 16, 16) [1x1 conv with stride=2]
        # This ensures the shortcut output matches the main path dimensions
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualCNN(nn.Module):
    """
    CNN with residual connections.
    """

    def __init__(self):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, 10)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # ResidualBlock handles the shape change, use it like a normal Conv2d
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # The first ResidualBlock handles the shape change,
        # the rest of the blocks just need to handle the channel change
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # Put 3 residual blocks in a row
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
