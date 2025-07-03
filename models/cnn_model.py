
# models/cnn_model.py
import torch.nn as nn

class TuringCNN(nn.Module):
    def __init__(self):
        super(TuringCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Output: Du, Dv, F, k
        )

    def forward(self, x):
        return self.model(x)


# Takes in a 128×128 grayscale pattern

# Passes it through 3 convolution blocks

# Outputs a 4D vector [Du, Dv, F, k]

# Fully connected layers let it learn complex patterns