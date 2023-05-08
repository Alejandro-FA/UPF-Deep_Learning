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
# X_train, X_test = data["X_train"], data["X_test"]
# Y_train, Y_test = data["Y_train"], data["Y_test"]

# print(f"Train sequences shape: {X_train.shape}") # (210, 40) (# sequences, # datapoints per sequence)
# print(f"Train labels shape: {Y_train.shape}") # Train Labels: (210,)  (# of sequences, )
# print(f"Test sequences shape: {X_test.shape}")
# print(f"Test labels shape: {Y_test.shape}")

# classes = np.unique(Y_train)
# n_classes = len(classes)
# colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

# fig, axs = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes))
# for class_idx in range(0, n_classes):
#     axs[class_idx].plot(X_test[Y_test == class_idx, :].T, "-o", c=colors[class_idx])
#     axs[class_idx].set_title(f"Tests Seqs. - Class {class_idx}")


# """
# Train a RNN or LSTM to solve the multi-class sequence classification problem.
#     -   Adapt the classification loss and the SequenceClassifier module
#     -   Adapt the test_sequence_classifier function to compute the multi-class accuracy and be able to visualize the confusion matrix
# """
# # Define module encapsulating a Sequence Classifier using RNN or LSTMs and setting different architecture hyper-parameters
# class SequenceClassifier(nn.Module):
#     def __init__(self, input_size: int = 1, hidden_size: int = 5, num_layers: int = 1, use_lstm: bool = False, bias: bool = False, n_classes: int = 2):
#         super().__init__()
#         self.use_lstm = use_lstm
        
#         if use_lstm:
#             self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bias=bias)
#         else:
#             self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bias=bias)
        
#         self.activation = nn.Softmax(dim=1) # FIXME: Change if necessary
#         self.last_linear = nn.Linear(hidden_size, n_classes)

#     def forward(self, X):
#         _, last_states = self.rnn(X)
#         # Get last hidden state for last layer. Ignore cell state in case of LSTMs
#         if not self.use_lstm:
#             last_hidden_state = last_states[-1, :, :].squeeze(0)
#         else:
#             last_hidden_state = last_states[0][-1, :, :].squeeze(0)
#         # Get sequence label probability using the last hidden state
#         output = self.activation(self.last_linear(last_hidden_state))
#         return output


# def train_sequence_classifier(X_train, Y_train, seq_classifier: SequenceClassifier, optimizer, loss_func, seed_value, epochs=100):
#     """
#     Given a dataset (with points and target labels), trains a SequenceClassifier.
#     """
#     torch.manual_seed(seed_value) # Ensures that all trainings use the same random numbers
#     loss_its = []
#     for iter in range(epochs):
#         optimizer.zero_grad()
#         output = seq_classifier(X_train)
#         loss = loss_func(output, Y_train)
#         loss_its.append(loss.item())
#         loss.backward()
#         optimizer.step()
    
#     print(f"Final loss: {loss.item()}")
#     return np.asarray(loss_its)


# def test_sequence_classifier(X_test, Y_test, seq_classifier):
#     """
#     Given a dataset (with points and target labels), computes the accuracy of the model.
#     """
#     output = seq_classifier(X_test)
#     classification = torch.argmax(output, dim=1)
#     conf_matrix = confusion_matrix(y_true=Y_test, y_pred=classification, normalize=None)
#     accuracy = (torch.sum(classification == Y_test) / output.shape[0]).item()

#     print(f"Test Accuracy: {np.round(accuracy * 100, decimals=3)} %")
#     return accuracy, conf_matrix


# def allocate(item, use_cuda=False):
#     if use_cuda:
#         return item.cuda()
#     else:
#         return item.cpu()


# """
# Experiment with different models by changing different hyper-parameters
# (e.g, num_layers, hidden_size, optimiziers, activation_functions for RNNs, etc..)
# and evaluate  the results for each of them on the testing set.
# """
# # Dataset to PyTorch format
# X_train_pt = allocate( torch.from_numpy(X_train).float().unsqueeze(2), use_cuda )
# Y_train_pt = allocate( torch.from_numpy(Y_train).long(), use_cuda )
# X_test_pt = allocate( torch.from_numpy(X_test).float().unsqueeze(2), use_cuda )
# Y_test_pt = allocate( torch.from_numpy(Y_test).long(), use_cuda )

# print('\nX_train and Y_train shape after being moved to torch:')
# print(X_train_pt.shape) # (210, 40, 1)
# print(Y_train_pt.shape) # (210, 1)
# print('-------------------------------')

# # Define Cross Entropy Loss
# loss_func = nn.CrossEntropyLoss()

# # Hyperparameters
# input_size = 1  # number of features of each point
# hidden_sizes_list = [5, 5, 20, 20, 100, 100]
# num_layers_list = [3, 10] * 3
# use_lstm_list = [True] * 6
# bias = False
# optimizer_class = torch.optim.Adam
# lr = 1e-3
# epochs = 100

# # Train the models
# losses_models = {}
# test_accuracy_models = {}
# confusion_matrix_models = {}

# for hidden_size, num_layers, use_lstm in zip(hidden_sizes_list, num_layers_list, use_lstm_list):
#     model_id = f"H{hidden_size}_NL{num_layers}_LSTM{int(use_lstm)}"
#     print(f"\nTraining: {model_id}")
    
#     seq_classifier = SequenceClassifier(input_size, hidden_size, num_layers, use_lstm, bias, n_classes)
#     allocate(seq_classifier, use_cuda)
#     optimizer = optimizer_class(seq_classifier.parameters(), lr=lr)
    
#     losses_models[model_id] = train_sequence_classifier(X_train_pt, Y_train_pt, seq_classifier, optimizer, loss_func, seed_value, epochs)
#     test_accuracy_models[model_id], confusion_matrix_models[model_id] = test_sequence_classifier(X_test_pt, Y_test_pt, seq_classifier)

# # Visualize loss function evolution
# fig2 = plt.figure(2)
# for model, loss in losses_models.items():
#     accuracy = np.round(test_accuracy_models[model] * 100, decimals=1)
#     plt.plot(loss, label=f"Train loss {model}, accuracy = {accuracy}")

# plt.title(f'Loss evolution', fontsize=14, fontweight="bold")
# plt.xlabel('Iterations')
# plt.ylabel('Loss Val')
# plt.ylim(0, None)
# plt.grid() 
# plt.legend()
# if save_figure: plt.savefig(f"{output_path}fig2.png", dpi=400)

# # Visualize the confusion matrices
# for model, cm in confusion_matrix_models.items():
#     fig = ConfusionMatrixDisplay(cm).plot()
#     fig.ax_.set_title(model)

# """
# Visualise, analyse and discuss the results in the report.
# """
# # TODO:

###############################################################################
#                              EXERCISE 2
###############################################################################
import pickle

with open(input_path+'training.pkl', 'rb') as pkl_file:
  train = pickle.load(pkl_file)

with open(input_path+'testing_corrupted.pkl', 'rb') as pkl_file:
  test = pickle.load(pkl_file)
vocabulary = [char for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-'] # Predefined vocabulary

# Corrupted Testing Sequence Visualization
def decode_message(message, vocabulary):
  decoded = ''
  for letter_code in message:
    decoded += vocabulary[letter_code]
  return decoded
  

# The data structure is 
# [
#   [encrypted0, decypted0]
#   [encrypted1, decypted1] 
# ]

### 1. Discover the keyword used to encrypt the plaintext data. (hint: use Vigenère table and some samples) manually?

encrypted1 = decode_message(train[0][0], vocabulary)
decrypted1 =  decode_message(train[0][1], vocabulary)

def find_vigenere_keyword(plaintext, ciphertext):
    plaintext = plaintext.upper()
    ciphertext = ciphertext.upper()

    keyword = ""
    for i in range(len(plaintext)):
        p = ord(plaintext[i]) - 65  # Convert plaintext character to a numeric value (0-25)
        c = ord(ciphertext[i]) - 65  # Convert ciphertext character to a numeric value (0-25)
        k = (c - p) % 26  # Calculate the shift between the plaintext and ciphertext characters
        keyword += chr(k + 65)  # Convert the shift value back to a character and append it to the keyword


    return keyword

keyword = find_vigenere_keyword(encrypted1, decrypted1)
print(f"The keyword used is: {keyword}")

### 2. Analyze the data distribution and define the new alphabet. Do plots (histogram)
train_cypher = [pair[0] for pair in train]
train_plaintext = [pair[1] for pair in train]
train_cypher = torch.cat(train_cypher)
train_plaintext = torch.cat(train_plaintext)

test_cypher = [pair[0] for pair in test]
test_plaintext = [pair[1] for pair in test]
test_cypher = torch.cat(test_cypher)
test_plaintext = torch.cat(test_plaintext)

bins = 27

fig3, axs = plt.subplots(2,2)
fig3.suptitle('Data Distribution')
axs[0, 0].hist(train_plaintext, bins, color = "blue")
axs[0, 0].set_title("Train plaintexts")
axs[0, 1].hist(train_cypher, bins, color = "orange")
axs[0, 1].set_title("Train cyphertexts")
axs[1, 0].hist(test_plaintext, bins, color = "blue")
axs[1, 0].set_title("Test plaintexts")
axs[1, 1].hist(test_cypher, bins, color = "orange")
axs[1, 1].set_title("Test cyphertexts")



### 3. Design/Implement an strategy to train the model so that it can be robust to missing characters in the testing samples ( to be discussed in class )

class DecrypterNetwork(nn.Module):
  def __init__(self,
               hidden_size : int = 8, 
               num_layers = 1,
               num_letters = 26,
               letters_embedding_size : int = 8,
               use_lstm : bool = False):
    # Define RNN or LSTM architecture
    super().__init__()
    self.hidden_size = hidden_size
    self.num_letters = num_letters
    self.letters_embedder = torch.nn.Embedding(num_letters, letters_embedding_size)
    self.use_lstm = use_lstm
    self.softmax = nn.Softmax(dim=1)
    if(use_lstm):
      self.rnn =  nn.LSTM(input_size = letters_embedding_size, hidden_size = hidden_size, 
                          num_layers=num_layers, batch_first = True)
    else:
      self.rnn =  nn.RNN(input_size = input_size, hidden_size = hidden_size, 
                         num_layers=num_layers, batch_first = True)
    self.last_linear = nn.Linear(hidden_size,num_letters)

  def forward(self, X):
    N = X.shape[0]
    L = X.shape[1]
    embedded_letters = self.letters_embedder(X)
    # Get hidden states for all letters in the sequence
    hidden_states,_ = self.rnn(embedded_letters)
    # In case of multiple input sequneces flat (N,L,hidden_size) to (N*L,hidden_size) for linear layer
    hidden_states_concat = hidden_states.reshape(-1,self.hidden_size) 
    # Get letters probability using the hidden states for each position in the sequence
    letters_loggits = self.last_linear(hidden_states_concat)
    #Use soft-max over logits and reshape to format (N,L,num_letteres)
    letters_probs = self.softmax(letters_loggits).reshape(N,L,self.num_letters)
    return letters_probs

def train_test(model, num_epochs, loss_fn, optimizer, 
               train_encrypted, train_decrypted, test_encrypted, test_decrypted,
               vocabulary, use_cuda = False):
    if(use_cuda):
      model = model.cuda()
      train_encrypted = train_encrypted.cuda()
      train_decrypted = train_decrypted.cuda()
      test_encrypted = test_encrypted.cuda()
      test_decrypted = test_decrypted.cuda()
    else:
      model = model
      train_encrypted = train_encrypted.cpu()
      train_decrypted = train_decrypted.cpu()
      test_encrypted = test_encrypted.cpu()
      test_decrypted = test_decrypted.cpu()

    accuracies, max_accuracy = [], 0
    loss_hist = []
    acc_hist = []
    epochs_hist = []
    test_loss_hist = []

    for epoch in range(num_epochs):
        # TRAINING AND BACK-PROPAGATION
        optimizer.zero_grad()
        letters_probs = model(train_encrypted)
        loss = loss_fn(letters_probs.log().permute(0,2,1), # rearrange as to (N_sequences, N_letters, N_lenght_sequences)
                       train_decrypted)
        loss.backward() # Backpropagate
        optimizer.step() # Update weights
        loss_hist.append(loss.item())

        # EVALUATION
        letters_probs = model(test_encrypted)
        test_loss = loss_fn(letters_probs.log().permute(0,2,1), # rearrange as to (N_sequences, N_letters, N_lenght_sequences)
                            test_decrypted)
        _,maxprob_letters_idx = letters_probs.max(dim=2) # get letter with maximum prob
        accuracy = ((maxprob_letters_idx==test_decrypted)*1.0).mean() # compute accuracy
        test_loss_hist.append(test_loss.item())
        acc_hist.append(accuracy.item())

        if(epoch%50==0):
          print(f'Epoch {epoch} \t Train Loss {round(loss.item(),3)} \t Test Loss {round(test_loss.item(),3)} \t Test Acc. (%)  {round(accuracy.item()*100,1)}')
    print(f'Final Epoch \t Train Loss {round(loss.item(),3)} \t Test Loss {round(test_loss.item(),3)} \t Test Acc. (%)  {round(accuracy.item()*100,1)}')
    return model,loss_hist, test_loss_hist, acc_hist
  

# Converting training and testing datasets into PyTorch Tensor (N_seqs, lenght_seqs)
train_encrypted = torch.concat([train_sample[0].unsqueeze(0) for train_sample in train],dim=0)
train_decrypted = torch.concat([train_sample[1].unsqueeze(0) for train_sample in train],dim=0)
test_encrypted = torch.concat([test_sample[0].unsqueeze(0) for test_sample in test_clean],dim=0)
test_decrypted = torch.concat([test_sample[1].unsqueeze(0) for test_sample in test_clean],dim=0)

# Initialize  Decrypter Network 
letters_embedding_size = 4
hidden_size = 16
num_letters = len(vocabulary)
num_layers = 1
use_lstm = True


decrypter_network = DecrypterNetwork(letters_embedding_size=letters_embedding_size,
                                    num_layers=num_layers,
                                    num_letters=num_letters,
                                    hidden_size=hidden_size,
                                    use_lstm=use_lstm)

# Define loss, optimizer and run training/evaluation loop
num_epochs=1000
CE_loss = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(decrypter_network.parameters(), lr=1e-2)

decrypter_network,loss_hist, test_loss_hist, acc_hist = train_test(decrypter_network, 
                                                 num_epochs, 
                                                 CE_loss, 
                                                 optimizer,
                                                 train_encrypted, train_decrypted, test_encrypted, test_decrypted,
                                                 vocabulary,
                                                 use_cuda=False)

plt.plot(loss_hist, '-.r', linewidth=1.0, label='train_loss')
plt.plot(test_loss_hist,'-b', linewidth=1.0, label='test_loss')
plt.xlabel('train step', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.legend()
plt.show()

plt.plot(acc_hist, linewidth=3.0, label='test_acc')
plt.xlabel('train step', fontsize=14)
plt.ylabel('accuracy(%)', fontsize=14)
plt.ylim([0, 1])
plt.xlim([0, num_epochs])
plt.legend()
plt.show()

decrypter_network = decrypter_network.cpu()
for idx_sample in range(0,3):
  # Inference over single training sequence
  letters_probs = decrypter_network(test[idx_sample][0].unsqueeze(0))
  # get index of letter with max probability 
  _,maxprob_letters_idx = letters_probs.max(dim=2)
  print('--------------------------------------')
  print(f'Original Message encrypted: {decode_message(test[idx_sample][0],vocabulary)}')
  print(f'Message decrypted: {decode_message(test[idx_sample][1],vocabulary)}')
  print(f'Prediction Message decrypted: {decode_message(maxprob_letters_idx[0],vocabulary)}')
  acc = (1.0*(maxprob_letters_idx[0]==test[idx_sample][1])).mean().item()
  print(f'Prediction Message Accuracy : {round(acc,2)}')

###############################################################################
#                          matplotlib magic
###############################################################################
plt.show()
