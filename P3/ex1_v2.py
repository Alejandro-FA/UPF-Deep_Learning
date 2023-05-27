import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.io as sio
from torch import nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import MyTorchWrapper as mtw
import time

# from google.colab import drive
# # Mount Google Drive
# drive.mount('/content/drive')
# %cd "/content/drive/DeepLearning/P3/"

data_path = 'P3/Data/'
results_path = 'P3/Results/'
run_train = True # Whether to train a model or not
run_test = True # Whether to test a model or not
device = mtw.get_torch_device(use_gpu=True, debug=True)
seed_value = 10
torch.manual_seed(seed_value)

"""
Instantiate IOManager to handle IO operations
"""
iomanager = mtw.IOManager(storage_dir=results_path + 'models/')
model_id = iomanager.next_id_available()

if run_test and not run_train:
    model_id = input("Introduce the model id that you want to test: ")
    while not iomanager.exists(model_id):
        print("Model does not exist")
        model_id = input("Introduce the model id that you want to test: ")


"""# Ex. 1
1. Try to obtain the maximum test accuracy possible in the SVHN dataset. For this purpose train/test different variants of the CNN provided in P3-Part1-Examples.
   You can explore different strategies:

- 1.1. Increase the size of the network by adding one ore more conv. layers. You can also increase the number of filters in each layer.

- 1.2. Try different optimizers such as Adam or SGD with momentum and modify the learning rate. You can check: https://pytorch.org/docs/stable/optim.html

- 1.3. Explore different random transformations during training ( Data augmentation ) such as random cropping with padding. You can check: https://pytorch.org/docs/stable/torchvision/transforms.html . Remember that these random transofrmation must not be used during testing.

- 1.4 Whatever you consider...


1. Save all the different models , compare their test accuracy and analyse the results. Discuss what model has been shown more effective and why have you used the different strategies.
"""

"""
Define the dataset class and the network architecture class
"""
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
    
    
    # Return the number of images
    def __len__(self):
        return self.data.shape[3]


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # Formula: output size = 1 + (n + 2p - f) / s
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=15, stride=1)
        # input: 3x32x32, output: 32x16x16
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1)
        # input: 32x16x16, output: 64x8x8
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        # input: 64x8x8, output: 128x4x4

        self.conv_2ndlast = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        # input: 128x4x4, output: 512x1x1
        
        self.conv_last = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        # input: 512x1x1, output: num_classes x 1x1

        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.pool1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.pool2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.pool3(out)
        out = self.relu(out)

        out = self.conv_2ndlast(out)
        out = self.avgpool(out)
        out = self.relu(out)

        out = self.conv_last(out)

        out = out.reshape(out.size(0), -1)
        # # print(out.shape)
        # out = out.reshape(out.size(0), -1) #128,32,8,8 -> 128,8*8*32
        # out = self.fc(out) # TODO:  we don't need sigmoid or other activation function at the end beacuse we will use nn.CrossEntropyLoss() (check documentation to understand why)
        
        return out


"""
Load data 
"""
tr = transforms.Compose([
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean = [.5], std = [.5])
        ])

SVHNTrain = SVHN(data_path+'/svhn/train_32x32.mat', tr)
SVHNTest = SVHN(data_path+'/svhn/test_32x32.mat', tr)

train_loader = torch.utils.data.DataLoader(dataset=SVHNTrain, batch_size=256, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=SVHNTest, batch_size=len(SVHNTest), pin_memory=True)


"""
Define hyperparameters
"""
CNN = ConvNet()
# optimizer = torch.optim.SGD(CNN.parameters(),lr = .001, weight_decay=1e-5, momentum=0.9)
epochs = 5
optimizer = torch.optim.Adam(CNN.parameters(), lr = .001)
evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss()) # Cross entropy loss for classification problems


"""
Train the model
"""
if run_train: # Train the model
    start_time = time.time()

    trainer = mtw.Trainer(CNN, evaluation=evaluation, epochs=epochs, optimizer=optimizer, data_loader=train_loader, device=device)
    trainer.seed_value = seed_value
    train_results = trainer.train()
    train_losses = train_results.loss
    train_accuracies = train_results.accuracy
    
    fig1 = plt.figure(1)
    plt.plot(train_accuracies)
    plt.title("Train accuracy at each step")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy (%)")
    plt.ylim((0, 100))
    plt.grid()

    fig2 = plt.figure(2)
    plt.plot(train_losses)
    plt.title("Loss at each step")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.ylim((0, None))
    plt.grid()

    # Save the model checkpoint
    iomanager.save(model=CNN, model_id=model_id)


"""
Test the model
"""
if run_test:
    iomanager.load(model=CNN, model_id=model_id)
    tester = mtw.Tester(model=CNN, evaluation=evaluation, data_loader=test_loader, device=device)
    test_results = tester.test()
    print(f'Test Accuracy of the model on the {len(SVHNTest)} test images: {test_results.accuracy} %')

    if run_train:
        # Save a model summary
        summary = mtw.training_summary(trainer, test_results)
        iomanager.save_summary(summary_content=summary, model_id=model_id)




# """# Ex. 2

# # **Building your own efficient Convolutional Network architecture for SVHN**

# In the following,  you will need to build your own CNN architecture to predict digit numbers on the SVHN dataset. You are completely free to use any kind of layers and hyper-parameters for this purpose. Your goal is to acheive the maximum possible accuracy on the test set (the better, the higher score you'll get in the exercice). The only constraint is that your model should not contain more than 150K parameters. Below, we provide a simple code to compute the number of parameters in a model.

# ## Computing model parameters
# """

# # Compute model paramters
# def compute_model_params(model):
#   params = 0
#   for p in model.parameters():
#     params+= p.numel()
#   return params

# import torch
# import torch.nn as nn

# # ResNet style network
# class ResSim(nn.Module):
#     def __init__(self, num_classes=10):
        
#         super(ResSim, self).__init__()
        
#         self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
#         self.conv21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
#         self.fc = nn.Linear(8*8*64, num_classes)
        
#         self.maxpool= nn.MaxPool2d(kernel_size=2, stride=2)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
        
#         out11 = self.relu(self.conv11(x))
#         out12 = self.relu(self.conv12(out11)) + out11

#         out = self.maxpool(out12)

#         out21 = self.relu(self.conv21(out)) 
#         out = self.relu(self.conv22(out21)) + out21
#         out = self.maxpool(out)
        
#         #print(out.shape)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
        
#         return out
# # Initialize the model
# model = ResSim(num_classes=10)
# # Compute and print number of params
# n_params = compute_model_params(model)
# print("ResNet Parameters: " + str(n_params)) ## 
# print("THIS MODEL CONTAINS 153K PARAMS, IT IS CONSIDERED NOT VALID FOR THE EXERCICE!!!!!!")

# '''
# 1. Design and implement your own CNN. Given that the number of parameters must be small, you can check some papers focused on efficient CNN architectures and get some ideas.
#   - MobileNet: https://arxiv.org/abs/1704.04861
#   - MobileNet V2: https://arxiv.org/pdf/1801.04381.pdf
#   - SqueezeNet: https://arxiv.org/abs/1602.07360
#   - ShuffleNet: https://arxiv.org/abs/1707.01083
#   - ESPNet V2: https://arxiv.org/abs/1811.11431
# 2. Train it and test it on SVHN using the provided code.
# 3. Discuss what approaches have you tried, why, and which ones have shown to be more beneficial.
# '''

# """## Sol. 2

# ### Define your own model and check the number of total parameters
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# #Mobilenet Style Architecture
# class MyModel(nn.Module):
#     def __init__(self, num_classes=10):
        
#         super(MyModel, self).__init__()
#         self.linear = nn.Linear(32*32*3,num_classes)
#         '''
#         REPLACE WITH YOUR CODE HERE
#         '''
                

                
#     def forward(self, x):
#         return self.linear(x.view(x.shape[0],-1))
#         '''
#         REPLACE WITH YOUR CODE HERE
#         '''


# model = MyModel(num_classes=10)
# n_params = compute_model_params(model)
# print("MyModel Parameters: " + str(n_params))

# """### Train your model on SVHN
# It is not allowed to change training hyper-parameters such as learning rate, batch size or number of epochs. You can only modify the architecture definition.
# """

# ## Create SVHN database

# # All the data will be loaded from the provided file in Data/mnist.t
# import torch 
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as tf
# import matplotlib.pyplot as plt
# from PIL import Image
# import scipy.io as sio
# from google.colab import drive
# import numpy as np

# # Mount Google Drive
# drive.mount('/content/drive')
# data_path = '/content/drive/My Drive/DeepLearning_2021/P4/Data/'
# results_path = '/content/drive/My Drive/DeepLearning_2021/P4/Results/'

# #Making native class loader
# class SVHN(torch.utils.data.Dataset):
#     # Initialization method for the dataset
#     def __init__(self,dataDir = data_path+'/svhn/train_32x32.mat',transform = None):
#         mat_loaded = sio.loadmat(dataDir)
#         self.data = mat_loaded['X']
#         self.labels = mat_loaded['y'].squeeze()
#         self.labels -= self.labels.min()
#         self.transform = transform
#      # What to do to load a single item in the dataset ( read image and label)    
#     def __getitem__(self, index):
#         data = self.data[:,:,:,index]
#         lbl = self.labels[index]
        
#         data = Image.fromarray(data)
#         # Apply a trasnformaiton to the image if it is indicated in the initalizer
#         if self.transform is not None : 
#             data = self.transform(data)
        
#         # return the image and the label
#         return data,lbl
    
#         pass
    
#     # Return the number of images
#     def __len__(self):
#         return self.data.shape[3]

# # Create train data loader
# tr = tf.Compose([
#         tf.ToTensor(), 
#         tf.Normalize(mean = [.5], std = [.5])
#         ])
# SVHNTrain = SVHN(data_path+'/svhn/train_32x32.mat',tr)
# train_loader = torch.utils.data.DataLoader(dataset=SVHNTrain,
#                                                batch_size=256, 
#                                                shuffle=True)

# # Train function
# def train(CNN, train_loader, optimizer, num_epochs=5, model_name='model.ckpt', device='cpu'):
#     CNN.train() # Set the model in train mode
#     total_step = len(train_loader)
#     losses_list = []
#     criterion = nn.CrossEntropyLoss()
#     # Iterate over epochs
#     for epoch in range(num_epochs):
#         # Iterate the dataset
#         loss_avg = 0
#         nBatches = 0
#         for i, (images, labels) in enumerate(train_loader):
#             # Get batch of samples and labels
#             images = images.to(device)
#             labels = labels.type(torch.LongTensor).to(device)

#             # Forward pass
#             outputs = CNN(images)
#             loss = criterion(outputs, labels)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_avg += loss.cpu().item()
#             nBatches+=1
#             if (i+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch+1, num_epochs, i+1, total_step, loss_avg / nBatches))
#         print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch+1, num_epochs, i+1, total_step, loss_avg / nBatches))
#         losses_list.append(loss_avg / nBatches)
#         torch.save(CNN.state_dict(), results_path+ '/' + model_name)
          
#     return losses_list 

# # Test funcion
# def test(CNN, test_loader):
#   with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             # get network predictions
#             outputs = CNN(images)

#             # get predicted class
#             _, predicted = torch.max(outputs.data, 1)

#             # compare with the ground-truth
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         # return accuracy
#         return 100 * correct / total

# #Train MyModel
# my_model = MyModel()

# #Initialize optimizer 
# learning_rate = .1
# optimizer = torch.optim.SGD(my_model.parameters(),lr = learning_rate, 
#                             weight_decay=1e-5, momentum=0.9)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# my_model = my_model.to(device)

# losses_list = train(my_model, train_loader, optimizer, num_epochs=10, model_name='my_net.ckpt', device=device)

# """### Test your model
# As a reference, 93% of accuracy can be easily achieved by using only ~55000 parameters.
# """

# # Show results for all the trained models
# SVHNTest = SVHN(data_path+'/svhn/test_32x32.mat',tr)
# test_loader = torch.utils.data.DataLoader(dataset=SVHNTest,
#                                                batch_size=256, 
#                                                shuffle=True)

# #
# my_net = MyModel()
# my_net.load_state_dict(torch.load(results_path + 'my_net.ckpt'))
# my_net.cuda()
# acc = test(my_net, test_loader)
# print('Accuracy MyNet: ' + str(acc))

# """# Ex. 3"""

# '''
# 1. Use the implemented architecture in the previous exercice to solve the transfer learning
#    task provided in the examples.
# 2. Try to fine-tune not only the last layer for the CNN but a larger subset of parameters.
# 2. Report the test accuracy in each case and discuss the results. 
# '''

# """## Sol. 3

# ### Initialize DataLoaders for Transfer Learning experiment
# """

# # Big dataset with numbers from 1 to 8
# SVHNTrain_TL = SVHN(data_path+'/svhn_tl/tl_train_32x32.mat',tr)
# tl_train_loader = torch.utils.data.DataLoader(dataset=SVHNTrain_TL,
#                                                batch_size=256, 
#                                                shuffle=True)

# # 200 samples of 0s and 9s
# SVHNTrain_TL_0_9 = SVHN(data_path+'/svhn_tl/tl_train_9_10_32x32.mat',tr)
# tl_train_loader_0_9 = torch.utils.data.DataLoader(dataset=SVHNTrain_TL_0_9,
#                                                   batch_size=64, 
#                                                   shuffle=True)
# # Test dataset with 0 and 9s
# SVHNTest_TL_0_9 = SVHN(data_path+'/svhn_tl/tl_test_9_10_32x32.mat',tr)
# tl_test_loader_0_9 = torch.utils.data.DataLoader(dataset=SVHNTest_TL_0_9,
#                                                   batch_size=64, 
#                                                   shuffle=True)

# """### Pre-train your model with the big dataset with numbers from 1 to 8"""

# #Train ResNet
# resnet_tl = MyModel(num_classes=8)
# #Initialize optimizer 
# learning_rate = .1
# optimizer = torch.optim.SGD(resnet_tl.parameters(),lr = learning_rate, weight_decay=1e-5, momentum=0.9)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# resnet_tl = resnet_tl.to(device)

# losses_it = train(resnet_tl, tl_train_loader, optimizer, num_epochs=10, model_name='tl_mynet_svhn.ckpt', device=device)

# """### Fine-tune the pretrained network with the small dataset of 9s and 0s"""

plt.show()