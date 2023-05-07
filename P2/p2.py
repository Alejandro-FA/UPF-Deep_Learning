import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

seed_value = 10
save_figure = False

###############################################################################
#                              EXERCISE 1
###############################################################################
"""
Load and visualize the training and testing data in 'data/P2_E1.npz'
"""
file_path = "P2/Data/P2_E1.npz"

data = np.load(file_path)
X_train, X_test = data["X_train"], data["X_test"]
Y_train, Y_test = data["Y_train"], data["Y_test"]

print(f"Train Seqs: {X_train.shape}")
print(f"Train Labels: {Y_train.shape}")
print(f"Test Seqs: {X_test.shape}")
print(f"Test Labels: {Y_test.shape}")

classes = np.unique(Y_train)
n_classes = len(classes)
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

fig, axs = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes))
for class_idx in range(0, n_classes):
    axs[class_idx].plot(X_test[Y_test == class_idx, :].T, "-o", c=colors[class_idx])
    axs[class_idx].set_title(f"Tests Seqs. - Class {class_idx}")

# Train Seqs: (210, 40)  (# sequences, # datapoints / sequence)
# Train Labels: (210,)  (# of sequences, )
# Test Seqs: (90, 40)
# Test Labels: (90,)


"""
Define the RNN Model
"""
# Define module encapsulating a Sequence Classifier using RNN or LSTMs and setting different architecture hyper-parameters
class SequenceClassifier(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 5, num_layers=1, use_lstm: bool = False):
        # Define RNN or LSTM architecture
        super().__init__()
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.activation = nn.Softmax() # FIXME: Change if necessary
        self.last_linear = nn.Linear(hidden_size, 1)

    def forward(self, X):
        _, last_states = self.rnn(X)
        # Get last hidden state for last layer. Ignore cell state in case of LSTMs
        if not self.use_lstm:
            last_hidden_state = last_states[-1, :, :].squeeze(0)
        else:
            last_hidden_state = last_states[0][-1, :, :].squeeze(0)
        # Get sequence label probability using the last hidden state
        output = self.activation(self.last_linear(last_hidden_state))
        return output


def train_sequence_classifier(X_train, Y_train, seq_classifier, optimizer, loss_func, epochs=100):
    loss_its = []
    
    for iter in range(epochs):
        optimizer.zero_grad()
        output = seq_classifier(X_train)
        if iter == 2:
            print("output:", output)
            print("Loss input shape: ", output.shape)
        loss = loss_func(output, Y_train)
        loss_its.append(loss.item())
        loss.backward()
        optimizer.step()
    
    print(f"Final loss: {loss.item()}")
    
    return np.asarray(loss_its)


# Compute accuracy accross testing dataset
def test_sequence_classifier(X_test, Y_test, seq_classifier, prob_threshold=0.5):
    output = seq_classifier(X_test)
    accuracy = (((output > prob_threshold) == (Y_test > prob_threshold)) * 1.0).mean()
    print(f"Test Accuracy: {accuracy.item()}")
    return accuracy.item()


"""
Train a RNN or LSTM to solve the multi-class sequence classification problem.
    -   Adapt the classification loss and the SequenceClassifier module
    -   Adapt the test_sequence_classifier function to compute the multi-class accuracy and be able to visualize the confusion matrix
"""
torch.manual_seed(seed_value)
# Dataset to PyTorch format
# X_train_pt = torch.from_numpy(X_train).float().unsqueeze(2).cuda()
# Y_train_pt = torch.from_numpy(Y_train).float().unsqueeze(1).cuda()
# X_test_pt = torch.from_numpy(X_test).float().unsqueeze(2).cuda()
# Y_test_pt = torch.from_numpy(Y_test).float().unsqueeze(1).cuda()
X_train_pt = torch.from_numpy(X_train).float().unsqueeze(2)
Y_train_pt = torch.from_numpy(Y_train).float().unsqueeze(1)
X_test_pt = torch.from_numpy(X_test).float().unsqueeze(2)
Y_test_pt = torch.from_numpy(Y_test).float().unsqueeze(1)

# Define Cross Entropy Loss
loss_func = nn.CrossEntropyLoss()

print(X_train_pt.shape) # (210, 40, 1)
print(Y_train_pt.shape) # (210, 1)

# Hyperparameters
input_size = 1  # number of features of each point
hidden_sizes_list = [5]
num_layers_list = [1]
use_lstm_list = [False]

# Train the models
losses_models = {}
test_accuracy_models = {}

for hidden_size, num_layers, use_lstm in zip(hidden_sizes_list, num_layers_list, use_lstm_list):
    model_id = f"H{hidden_size}_NL{num_layers}_LSTM{int(use_lstm)}"
    print(f"Training: {model_id}")
    
    seq_classifier = SequenceClassifier(input_size=input_size, use_lstm=use_lstm, num_layers=num_layers, hidden_size=hidden_size)
    # seq_classifier.cuda()
    
    optimizer = torch.optim.Adam(seq_classifier.parameters(), lr=1e-3)
    
    losses_models[model_id] = train_sequence_classifier(X_train_pt, Y_train_pt, seq_classifier, optimizer, loss_func, epochs=1000)
    test_accuracy_models[model_id] = test_sequence_classifier(X_test_pt, Y_test_pt, seq_classifier)


# Visualize loss function evolution
fig2 = plt.figure(2)
plt.title('Loss evolution', fontsize=14, fontweight="bold")
plt.plot(losses_models["H5_NL1_LSTM0"], color="black", label="Train loss")
plt.xlabel('Iterations (x100)')
plt.ylabel('Loss Val')
plt.legend()
plt.ylim(0, None)
plt.grid()

"""
Experiment with different models by changing different hyper-parameters
(e.g, num_layers, hidden_size, optimiziers, activation_functions for RNNs, etc..)
and evaluate  the results for each of them on the testing set.
"""


"""
Visualise, analyse and discuss the results in the report.
"""
# TODO:

###############################################################################
#                              EXERCISE 2
###############################################################################


###############################################################################
#                          matplotlib magic
###############################################################################
plt.show()
