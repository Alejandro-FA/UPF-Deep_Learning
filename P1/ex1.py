import pandas as pd 
import matplotlib.pyplot as plt #  matplotlib for visualization
import numpy as np


data_path = 'P1/Data/'
results_path = 'P1/Results'

# Load training and testing data
train_data = pd.read_csv(data_path + "train.csv")
test_data = pd.read_csv(data_path + "test.csv")

X_train = train_data[["X", "Y"]].to_numpy()
C_train = train_data["C"].to_numpy().reshape(100, 1)


# Visualize data
plt.title("Training data", fontsize=14, fontweight="bold")
plt.scatter(train_data.X, train_data.Y, c=train_data.C)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Normalize dataset
X_train = X_train / np.amax(X_train, axis=0)

# MLP Class:
class MLP(object):
    def __init__(self,inputNode=2,hiddenNode = 3, outputNode=1):        
        #Define MLP hyper-parameters
        self.inputLayerSize = inputNode # number of input dimensions (x)
        self.outputLayerSize = outputNode # number of output dimensions (y)
        self.hiddenLayerSize = hiddenNode # Dimension of intermediate layer (W_2)
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Compute inputs from outputs
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    # Derivative of sigmoid and loss with respect their inputs
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def loss(self, yHat, y):
        #Compute loss for given X,y, use weights already stored in class.
        J = 0.5*sum((y-yHat)**2)
        return J
        
    # Derivative of parameters
    def backward(self,X, yHat, y):
        self.yHat = yHat
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        return dJdW1,dJdW2

def train(NN,X, y,epoch = 10000,lr = .1):
    list_loss = []
    
    for i in range(epoch):
        #Compute derivative with respect to W and W2 for a given X and y:
        yHat = NN.forward(X)
        
        gradW1,gradW2 = NN.backward(X,yHat,y)
        #now update the weight using gradient descent
        NN.W1 -= gradW1 * lr 
        NN.W2 -= gradW2 * lr
        
        if i%100 == 0 : 
            loss = NN.loss(yHat,y)
            print('Loss {}={}'.format(i,loss))
            list_loss.append(loss)
        
    return list_loss


NN = MLP(hiddenNode=10)
#Train network with the data:
list_loss = train(NN, X_train, C_train, epoch=10000, lr=0.01)

# Loss function iterations
plt.plot(list_loss)
plt.xlabel('Iterations')
plt.ylabel('Loss Val')
plt.show()
