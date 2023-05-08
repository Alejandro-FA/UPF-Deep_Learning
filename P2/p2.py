import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

output_path = 'P2/Results/'
input_path = 'P2/Data/'
seed_value = 10
save_figure = True
use_cuda = False

###############################################################################
#                              EXERCISE 1
###############################################################################
"""
Load and visualize the training and testing data in 'data/P2_E1.npz'
"""
data = np.load(input_path + 'P2_E1.npz')
X_train, X_test = data["X_train"], data["X_test"]
Y_train, Y_test = data["Y_train"], data["Y_test"]

print(f"Train sequences shape: {X_train.shape}") # (210, 40) (# sequences, # datapoints per sequence)
print(f"Train labels shape: {Y_train.shape}") # Train Labels: (210,)  (# of sequences, )
print(f"Test sequences shape: {X_test.shape}")
print(f"Test labels shape: {Y_test.shape}")

classes = np.unique(Y_train)
n_classes = len(classes)
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

fig, axs = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes))
for class_idx in range(0, n_classes):
    axs[class_idx].plot(X_test[Y_test == class_idx, :].T, "-o", c=colors[class_idx])
    axs[class_idx].set_title(f"Tests Seqs. - Class {class_idx}")


"""
Train a RNN or LSTM to solve the multi-class sequence classification problem.
    -   Adapt the classification loss and the SequenceClassifier module
    -   Adapt the test_sequence_classifier function to compute the multi-class accuracy and be able to visualize the confusion matrix
"""
# Define module encapsulating a Sequence Classifier using RNN or LSTMs and setting different architecture hyper-parameters
class SequenceClassifier(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 5, num_layers: int = 1, use_lstm: bool = False, bias: bool = False, n_classes: int = 2):
        super().__init__()
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bias=bias)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bias=bias)
        
        self.activation = nn.Softmax(dim=1) # FIXME: Change if necessary
        self.last_linear = nn.Linear(hidden_size, n_classes)

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


def train_sequence_classifier(X_train, Y_train, seq_classifier: SequenceClassifier, optimizer, loss_func, seed_value, epochs=100):
    """
    Given a dataset (with points and target labels), trains a SequenceClassifier.
    """
    torch.manual_seed(seed_value) # Ensures that all trainings use the same random numbers
    loss_its = []
    for iter in range(epochs):
        optimizer.zero_grad()
        output = seq_classifier(X_train)
        loss = loss_func(output, Y_train)
        loss_its.append(loss.item())
        loss.backward()
        optimizer.step()
    
    print(f"Final loss: {loss.item()}")
    return np.asarray(loss_its)


def test_sequence_classifier(X_test, Y_test, seq_classifier):
    """
    Given a dataset (with points and target labels), computes the accuracy of the model.
    """
    output = seq_classifier(X_test)
    classification = torch.argmax(output, dim=1)
    conf_matrix = confusion_matrix(y_true=Y_test, y_pred=classification, normalize=None)
    accuracy = (torch.sum(classification == Y_test) / output.shape[0]).item()

    print(f"Test Accuracy: {np.round(accuracy * 100, decimals=3)} %")
    return accuracy, conf_matrix


def allocate(item, use_cuda=False):
    if use_cuda:
        return item.cuda()
    else:
        return item.cpu()


"""
Experiment with different models by changing different hyper-parameters
(e.g, num_layers, hidden_size, optimiziers, activation_functions for RNNs, etc..)
and evaluate  the results for each of them on the testing set.
"""
# Dataset to PyTorch format
X_train_pt = allocate( torch.from_numpy(X_train).float().unsqueeze(2), use_cuda )
Y_train_pt = allocate( torch.from_numpy(Y_train).long(), use_cuda )
X_test_pt = allocate( torch.from_numpy(X_test).float().unsqueeze(2), use_cuda )
Y_test_pt = allocate( torch.from_numpy(Y_test).long(), use_cuda )

print('\nX_train and Y_train shape after being moved to torch:')
print(X_train_pt.shape) # (210, 40, 1)
print(Y_train_pt.shape) # (210, 1)
print('-------------------------------')

# Define Cross Entropy Loss
loss_func = nn.CrossEntropyLoss()

# Hyperparameters
input_size = 1  # number of features of each point
hidden_sizes_list = [5, 5, 20, 20, 100, 100]
num_layers_list = [3, 10] * 3
use_lstm_list = [True] * 6
bias = False
optimizer_class = torch.optim.Adam
lr = 1e-3
epochs = 100

# Train the models
losses_models = {}
test_accuracy_models = {}
confusion_matrix_models = {}

for hidden_size, num_layers, use_lstm in zip(hidden_sizes_list, num_layers_list, use_lstm_list):
    model_id = f"H{hidden_size}_NL{num_layers}_LSTM{int(use_lstm)}"
    print(f"\nTraining: {model_id}")
    
    seq_classifier = SequenceClassifier(input_size, hidden_size, num_layers, use_lstm, bias, n_classes)
    allocate(seq_classifier, use_cuda)
    optimizer = optimizer_class(seq_classifier.parameters(), lr=lr)
    
    losses_models[model_id] = train_sequence_classifier(X_train_pt, Y_train_pt, seq_classifier, optimizer, loss_func, seed_value, epochs)
    test_accuracy_models[model_id], confusion_matrix_models[model_id] = test_sequence_classifier(X_test_pt, Y_test_pt, seq_classifier)

# Visualize loss function evolution
fig2 = plt.figure(2)
for model, loss in losses_models.items():
    accuracy = np.round(test_accuracy_models[model] * 100, decimals=1)
    plt.plot(loss, label=f"Train loss {model}, accuracy = {accuracy}")

plt.title(f'Loss evolution', fontsize=14, fontweight="bold")
plt.xlabel('Iterations')
plt.ylabel('Loss Val')
plt.ylim(0, None)
plt.grid() 
plt.legend()
if save_figure: plt.savefig(f"{output_path}fig2.png", dpi=400)

# Visualize the confusion matrices
for model, cm in confusion_matrix_models.items():
    ConfusionMatrixDisplay(cm).plot()

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
