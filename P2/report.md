# Deep Learning Practice 2 report document

*RNNs and LSTMs*

Group 8

Andreu Garcies (240618), Alejandro Fernández (242349), Marc Aguilar (242192)

# Exercise 1

## Train a RNN or LSTM to solve the multi-class sequence classification problem.

### How have we adapted the classification loss and the SequenceClassifier module
In order to use the `SequenceClassifier` to solve our problem we modified the last linear layer and the activation function. 

First of all, after analyzing the shape of the data and the goal of our network we realized that in the last layer we needed at least as many neurons as classes we wanted to distinguish. Therefore, we added the parameter `n_classes` to both the class and the layer (`nn.Linear(hidden_size, n_classes)`). 

After the first modification and taking into account that we are solving a multiclass classification problem, the activation function which fitted the best was a **Softmax**, which directly returns probability that the sequence belongs to each of the classes. 

### How have we adapted the test_sequence_classifier function to compute the multi-class accuracy

For the confusion matrix we have decided to use the `sklearn.metrics`, in particular the `confusion_matrix` and `ConfusionMatrixDisplay` modules. We use the first module to compute the values of the confusion matrix and the second one to plot the results. We have not modified the `test_sequence_classifier` function in any other meaningful way. One minor detail to remark is that any operation related to confusion matrices cannot be performed in the GPU.


## Experiment with different models by changing different hyper-parameters (e.g, num_layers, hidden_size, optimiziers, activation_functions for RNNs, etc..) and evaluate the results for each of them on the testing set.

In order to search for the best performance possible we have tried multiple hyperparameter combinations, as well as some other alternatives like using bidirectional LSTMs and weight decay. Here we detail our exploration of several parameters and the results found:

- **Bias**: Using a bias parameter in the model consistently hurts performance, so it was rapidly discarded as a good option.

- **LSTM vs normal RNN**: Using LSTM instead of normal RNN is almost non-negotiable. During the hyperparameter exploration phase it has provided much better results in a consistent manner. So, after some time, we stopped trying normal RNNs.

- **Optimizer used**: For this model we tried multiple optimizers from those available in `pytorch` (in fact, we tried all of them). Although we did not spend time fine-tuning each of the optimizers, we saw that the one that gave us the best immediate was `Adam`, so we continued with it.

- **Number of layers**: After changing the number of layers in combination with multiple other parameters, we found that the best results were consistently achieved with 4 layers.

From this point onwards we present the hyperparameters that do not have as clear of a "best" value as the ones presented above.

- **Dropout**: At the beginning of the training process, we observed very inconsistent results. Minor changes in the hyperparameters produced very different results. To solve this, we tried to use dropout, which proved to be a great idea. A dropout around `0.05` increases the robustness of the model a lot, and it also improves the accuracy obtained in the test dataset. 

- **Learning rate ($\alpha$)**: We also tried multiple values for the learning rate. We found that any value above `1e-3` resulted in poor results, and anything below was too slow to train without any added benefit.

- **Hidden size**: The size of the hidden layers does not seem to be as important as the other parameters. In general, it seems that values outside the `20-100` range start to lose performance, but the final result depends on the machine that is executing the code.

Other configurations tried that were removed in the end:

- **Bidirectional LSTM**: When we heard about bidirectional LSTMs, we thought that they were a perfect fit for our case (since a sinusoid can be "predicted" both when seeing from left to right as from right to left). However, when we implemented it with pytorch we actually observed a reduction in performance. We believe that with more time invested it could have potentially improved the results that we currently have.

- **Weight decay**: Another overfitting countermeasure that we tried was weight decay, which did not provide any valuable results. Dropout worked consistently better. Furthermore, weight decay in combination with dropout also reduced the final accuracy obtained by the model. So we decided to not use it at all.

The following image shows the loss evolution for the models with all the different hyperparameters that we tested:

<img src="Results/fig2.png" style="zoom:30%"></img>

From the plot we can summarize what we have presented in this section: the best results correspond to the model with `4` hidden layers, of `100` neurons each and $\alpha$ = `0.005`.


# Exercise 2

## Discover the keyword & Verify it is the correct one

In order to discover the **keyword** we designed a very simple algorithm which computes the shift between both the plaintext and the cyphertext character by character. After running the algorithm in one pair of cypher and plaintext we discovered that the **keyword is PYTORCHPYTORCHPYTORCHPYTORCHPYTO**. We verified that encrypting with this keyword resulted in the same cyphertext for some of the pairs in the `test` dataset.

## Analyze the new dataset & Define the new alphabet

We can observe that the distribution of both **train and test** plaintexts is similar. This is what we expected since none of them are corrupted, and they are both in English so they should follow a similar distribution. On the other hand, in the test cyphertext dataset we can observe how a **new character** arises ("-"), and is the most frequent one. The train cyphertext character distribution has no anomalies. 

The construction of the new alphabet was quite straightforward since we just had to add the new character ("-") to the abecedary.  

## Design/Implement a strategy to train the model so that it can be robust to missing characters in the testing samples

The first time that we executed the model, we obtained the following results

<span style="color:red">**TO BE COMPLETED**</span>

Even though it may seem like the model is performing significantly well (with an accuracy of $\approx 88\%$) we need to take the following into account: *$12.5\%$ of the sentences of the testing dataset have been corrupted*. This means that the model is facing some data that it has not seen during the training step, thus, it is very likely that it does not know how to handle it. Indeed, if we separate the accuracy in corrupted characters vs non-corrupted characters that the model has a very poor performance when it comes to predict the plaintext character of a corrupted character `-`.

<span style="color:red">**INSERT IMAGE WITH THE DEFAULT RESULTS**</span>

As we can see, the model has a considerable margin of improvement. In this section, we will explain the different strategies that we implemented to overcome this problem of missclassifying corrupted characters and we are going to see how much of an impact they had in the overall performance of the model. 

### Data augmentation

First of all we thought that the most straightforward approach was augmenting our initial training dataset by **corrupting it several times** with different corruption rates. After some tests we did not achieve good results, so we augmented our dataset by gathering new data. 

We downloaded and parsed an English dictionary with a custom script. Then we extracted some definitions which matched our target length (32 characters). At the end we ended up with a new dataset of around **63K new sequences**, which can be found at `Data/our_training.pkl`. 

While we were doing hyperparameter modifications in order to improve the performance of the model, we started running out of GPU memory. To handle this issue we started using just a portion of the whole train dataset. Finally, we went back to our first approach since it is the one which gives us the best results.

<span style="color:red">Canviar si ens funciona amb el dataset propi, queda més pro</span>


### "Over-corruption" of the dataset

The training dataset does not come corrupted by default, so it is clear that we have to introduce hyphens in order to have a vocabulary that matches the one of the testing dataset. The question was, how much should we corrupt our training data?

As suggested in class, training the model with a harder task can help it to generalise better. During training we were able to confirm that this was indeed true. After testing some values, we found that a corruption ratio of `30%` provided quite better results than `12.5%` (which is the corruption ratio of the testing dataset).

As a note for reference, we tried with a corruption of `40%` and the model was not able to learn.

### Bidirectional LSTM

In order to predict which character is missing in a sequence, it seems obvious that knowing information about the past characters but also of the future ones is more helpful than just having the previous ones. So, this is the reason why we implemented the bidirectional LSTM. 

In order to implement it we just set the flag `bidirectional_lstm` to `True` and we had to duplicate the sizes of the hidden states and of the last linear layer. We had to do this because now we had the "forward" LSTMs and the "backward" ones. 

### Dropout

Given the good results obtained in the first exercise, we have decided to also implement dropout for this case. Once again, it appears to have very beneficial effects.
