import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch
import torchvision as T
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from os.path import exists
import re



data_path = 'P3/Data/'
results_path = 'P3/Results/'
seed_value = 10

torch.manual_seed(seed_value)

run_train = True
run_test = True

# Setting the model name
if run_train:
    files = listdir(results_path + "models/")
    models = list(filter(lambda name: ".ckpt" in name, files))
    pattern = re.compile("model_(\d+).ckpt")
    indices = [int(pattern.search(model).group(1)) for model in models]
    if len(models) == 0:
        indices = [0]
    train_model_name = "model_" + str(max(indices) + 1)
    test_model_name = train_model_name
    print(f"Training model {train_model_name}")


if run_test and not run_train:
    test_model_name = input("Introduce the model name that you want to test: ")
    while not exists(results_path + "models/" + test_model_name + ".ckpt"):
        print("Model does not exist")
        test_model_name = input("Introduce the model name that you want to test: ")


## Tests

# Define an standard CNN -> Two conv. blocks and linear layer 
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5,  padding=2)
        #input : 3 channel, output 16 channel, filter size : 5x5
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,  padding=1)
        #input : 16 channel, output 32 channel, filter size : 3x3

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        #input : 32 channel, output 64 channel, filter size : 3x3

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)
        #input : 64 channel, output 128 channel, filter size : 3x3
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

        
        self.maxpool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # print(out.shape)
        out = out.reshape(out.size(0), -1) #128,32,8,8 -> 128,8*8*32
        # print(out.shape)
        out = self.fc1(out) 
        out = self.relu(out)
        out = self.fc2(out) # we don't need sigmoid or other activation function at the end beacuse we will use nn.CrossEntropyLoss() (check documentation to understand why)

        return out 


CNN = ConvNet()
CNN = CNN.cpu()

# Cross entropy loss for classification problems
criterion = nn.CrossEntropyLoss()

#Initialize optimizer 
learning_rate = .001
optimizer = torch.optim.Adam(CNN.parameters(), lr = learning_rate)

# Device configuration (choose GPU if it is available )
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5

# dataloader
#Making native class loader
class SVHN(torch.utils.data.Dataset):
    # Initialization method for the dataset
    def __init__(self,dataDir = data_path+'/svhn/train_32x32.mat',transform = None):
        mat_loaded = sio.loadmat(dataDir)
        self.data = mat_loaded['X']
        self.labels = mat_loaded['y'].squeeze()
        self.labels -= self.labels.min()
        self.transform = transform
     # What to do to load a single item in the dataset ( read image and label)    
    def __getitem__(self, index):
        data = self.data[:,:,:,index]
        lbl = self.labels[index]
        
        data = Image.fromarray(data)
        # Apply a trasnformaiton to the image if it is indicated in the initalizer
        if self.transform is not None : 
            data = self.transform(data)
        
        # return the image and the label
        return data,lbl
    
        pass
    
    # Return the number of images
    def __len__(self):
        return self.data.shape[3]

tr = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomRotation(degrees=(-30,30)),
        transforms.Normalize(mean = [.5], std = [.5])
        ])

SVHNTrain = SVHN(data_path+'/svhn/train_32x32.mat', tr)


# network 
train_loader = torch.utils.data.DataLoader(dataset=SVHNTrain,
                                               batch_size=256, 
                                               shuffle=True)

# Train and save the model
if run_train:
    print(f"Training with {len(SVHNTrain)} images")
    CNN.train() # Set the model in train mode
    total_step = len(train_loader)
    accuracies = []
    train_losses = []
    # Iterate over epochs
    for epoch in range(num_epochs):
        # Iterate the dataset
        # total = 0
        # correct = 0
        for i, (images, labels) in enumerate(train_loader):
            # Get batch of samples and labels
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = CNN(images)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())        

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # with torch.no_grad():
            #     _, predicted = torch.max(outputs.data, 1)

            #     # compare with the ground-truth
            #     total += labels.size(0)
            #     correct += (predicted == labels).sum().item()
            
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        # accuracies.append(100 * correct / total)
        
        # print(f"\nEpoch {epoch+1}:")
        # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%\n' 
        #         .format(epoch+1, num_epochs, i+1, total_step, loss.item(), 100 * correct / total))
        

    with open(results_path + "models/" + train_model_name + ".txt", "w") as results_txt:
        results_txt.write(f"Train loss: {train_losses}\n")
        results_txt.write(f"Train accuracy:\n{accuracies}\n")


    # Save the model checkpoint
    torch.save(CNN.state_dict(), results_path + "models/" + train_model_name + ".ckpt")


if run_test:
    CNN.load_state_dict(torch.load(results_path + "models/" + test_model_name + ".ckpt"))
    # Load test dataset
    SVHNTest = SVHN(data_path+'/svhn/test_32x32.mat', tr)
    test_loader = torch.utils.data.DataLoader(dataset=SVHNTest,
                                               batch_size=256, 
                                               shuffle=True)
    CNN.eval() # Set the model in evaluation mode
    
    # Compute testing accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # get network predictions
            outputs = CNN(images)

            # get predicted class
            _, predicted = torch.max(outputs.data, 1)

            # compare with the ground-truth
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print('Test Accuracy of the model on the {} test images: {} %'.format(len(SVHNTest), 100 * correct / total))

        with open(results_path + "models/" + test_model_name + ".txt", "a") as results_txt:
            results_txt.write(f"Test accuracy: {100 * correct / total}\n")

with open(results_path + "models/" + test_model_name + ".txt", "a") as results_txt:
    results_txt.write(f"Epochs: {num_epochs}\n")
    results_txt.write(f"Loss function: {criterion}\n")
    results_txt.write(f"Optimizer: {optimizer}\n")
