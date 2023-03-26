import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
torch.manual_seed(123)
# Contrução do Modelo
class classificador(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) 
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()

        # output = (input - filter + 1) / stride
        # Convolução 1 -> (64 - 3 + 1) / 1 = 62x62
        # Pooling 1 -> Só dividir pelo kernel_size = 31x31
        # Convolução 2 -> (31 - 3 + 1)/ 1 = 29x29
        # Pooling 2 -> Só dividir pelo kernel_size = 14x14
        # 14 * 14 * 64
        # 33907200 valores -> 256 neurônios da camada oculta
        self.linear1 = nn.Linear(in_features=14*14*64, out_features=256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 4)

    def forward(self, X):
        X = self.pool(self.bnorm(self.activation(self.conv1(X))))
        X = self.pool(self.bnorm(self.activation(self.conv2(X))))
        X = self.flatten(X)

        # Camadas densas
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))
        
        # Saída
        X = self.output(X)

        return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

classificadorLoaded = classificador()
state_dict = torch.load('checkpoint.pth')
classificadorLoaded.load_state_dict(state_dict)
from PIL import Image
imagem_teste = Image.open("archive/Training/pituitary/Tr-pi_0010.jpg")

import numpy as np
imagem_teste = imagem_teste.resize((64, 64))
imagem_teste = imagem_teste.convert('RGB') 
imagem_teste = np.array(imagem_teste.getdata()).reshape(*imagem_teste.size, -1)
imagem_teste = imagem_teste / 255
imagem_teste = imagem_teste.transpose(2, 0, 1)
imagem_teste = torch.tensor(imagem_teste, dtype=torch.float).view(-1, *imagem_teste.shape)

classificadorLoaded.eval()
imagem_teste = imagem_teste.to(device)
output = classificadorLoaded.forward(imagem_teste)
output = F.softmax(output, dim=1)
top_p, top_class = output.topk(k = 1, dim = 1)
output = output.detach().numpy()
resultado = np.argmax(output[0])

print(resultado)