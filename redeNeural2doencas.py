import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(123)

#Base de dados
data_dir_train = './train'
data_dir_test = './test'

transform_train = transforms.Compose(
    [
      transforms.Resize([64,64]),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ]
)

transform_test = transforms.Compose(
    [
     transforms.Resize([64,64]),
     transforms.ToTensor()
    ]
)

train_dataset = datasets.ImageFolder(data_dir_train, transform = transform_train)
test_dataset = datasets.ImageFolder(data_dir_test, transform = transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle=True)
# Contrução do Modelo
classificador = nn.Sequential(
                              nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3),
                              nn.ReLU(),
                              nn.MaxPool2d(kernel_size = 2),
                              nn.Conv2d(64, 64, 3),
                              nn.ReLU(),
                              nn.MaxPool2d(2),
                              nn.Flatten(),
                              nn.Linear(in_features = 14*14*64, out_features = 4),
                              nn.ReLU(),
                              nn.Linear(4, 4),
                              nn.ReLU(),
                              nn.Linear(4, 4),
                              nn.Sigmoid()
                              )

criterion = nn.BCELoss()
optimizer = optim.Adam(classificador.parameters())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device
classificador.to(device)

def training_loop(loader, epoch):
    running_loss = 0.
    running_accuracy = 0.
    
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()        

        outputs = classificador(inputs)
                
        loss = criterion(outputs, labels.float().view(*outputs.shape))
        running_loss += loss.item()
        loss.backward()
        
        optimizer.step()

        predicted = torch.tensor([1 if output > 0.5 else 0 for output in outputs]).to(device)
        equals = predicted == labels.view(*predicted.shape)
        accuracy = torch.mean(equals.float())
        running_accuracy += accuracy
      
        # Imprimindo os dados referentes a esse loop
        print('\rÉpoca {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisão {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy), end = '\r')
        
    # Imprimindo os dados referentes a essa época
    print('\rÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss/len(loader), running_accuracy/len(loader)))

for epoch in range(100):
    print('Treinando...')
    training_loop(train_loader, epoch)
    classificador.eval()
    print("Validando")
    training_loop(test_loader, epoch)
    classificador.train()
 
classificador.eval()
torch.save(classificador.state_dict(), "checkpoint.pth")

