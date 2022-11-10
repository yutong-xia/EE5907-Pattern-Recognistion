import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

def get_dataloader(train_x, train_y, test_x, test_y, batch_size):
    train_x = Tensor(train_x)
    train_y= F.one_hot(Tensor(train_y - 1).long())
    test_x= Tensor(test_x)
    # test_y= F.one_hot(Tensor(test_y - 1).long())
    test_y= Tensor(test_y - 1)
    dataset_train = TensorDataset(train_x, train_y)
    dataset_test = TensorDataset(test_x, test_y)
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size, shuffle=False)
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, 
                               kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, 
                               kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(50 * 6 * 6, 500)
        self.out = nn.Linear(500, 26)

    def forward(self, input):
        output = F.relu(self.conv1(input.unsqueeze(1)))
        output = self.pool(self.bn1(output))  
        output = F.relu(self.conv2(output))
        output = self.pool(self.bn2(output) )
        output = output.view(-1, 50 * 6 * 6)
        output = F.relu(self.fc1(output))
        output = F.relu(self.out(output))
        return output
    
class CNN_trainer():
    def __init__(self, model, num_epochs = 100):
        super(CNN_trainer, self).__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.clip_grad_value = 5.0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = []
        self.acc = []
        
    def train(self, train_loader, num_epochs, test_loader):

        self.model.to(self.device)
        self.num_epochs = num_epochs
        max_acc = 0.0
        for epoch in range(num_epochs): 
            self.model.train()
            train_losses = []
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_value)
                self.optimizer.step()

                train_losses.append(loss.item())
                
            test_acc = self.test(test_loader)
            
            print('Epoch [{}/{}]  train_loss: {:.4f} test_acc: {:.4f}'.format(epoch + 1, num_epochs, np.mean(train_losses), test_acc))
            if test_acc > max_acc:
                max_acc = test_acc
                best_epoch = epoch + 1
                self.save_model()
            self.loss.append(np.mean(train_losses))
            self.acc.append(test_acc)
        self.best_epoch = best_epoch
        np.save('./results/cnn_loss.npy', np.array(self.loss))
        np.save('./results/cnn_acc.npy', np.array(self.acc))
        print('Epoc {}: Best acc {}.'.format(best_epoch, max_acc))


    def test(self, test_loader):
        
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        
        accuracy = 100 * accuracy / total
        return accuracy

    def save_model(self):
        path = "./results/cnn.pth"
        torch.save(self.model.state_dict(), path)
        
    def plot_curve(self):

        plt.figure(dpi=300,figsize=(8,6))
        plt.subplot(211)

        plt.plot(range(self.num_epochs), self.loss, label = 'Train Loss')

        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('(a) Train Loss', y = -0.3)
        plt.vlines(self.best_epoch, 0, 3, colors = "#d37a7d", linestyles = "dashed")
        plt.legend()

        plt.subplot(212)
        plt.plot(range(self.num_epochs), self.acc, label = 'Test Accuracy')

        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('(b) Test Accuracy', y = -0.3)
        plt.vlines(self.best_epoch, 0, 100, colors = "#d37a7d", linestyles = "dashed")
        plt.legend()

        plt.tight_layout()
        plt.savefig('./results/cnn_fig.pdf', bbox_inches='tight')
        
