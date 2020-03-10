# Neural network architecture similar to Eric's
# The only difference is in the activation functions

class AE_3D(nn.Module):
    def __init__(self, n_features=4):
        super(AE_3D, self).__init__()
        self.en1 = nn.Linear(n_features, 100)
        self.en2 = nn.Linear(100, 100)
        self.en3 = nn.Linear(100, 50)
        self.en4 = nn.Linear(50, 3)
        self.de1 = nn.Linear(3, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 100)
        self.de4 = nn.Linear(100, n_features)
        self.sigmoid = nn.Sigmoid()
    def encode(self, x):
        return self.en4(self.sigmoid(self.en3(self.sigmoid(self.en2(self.sigmoid(self.en1(x)))))))

    def decode(self, x):
        return self.de4(self.sigmoid(self.de3(self.sigmoid(self.de2(self.sigmoid(self.de1(self.sigmoid(x))))))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z).float()
