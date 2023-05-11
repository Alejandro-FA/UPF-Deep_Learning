# Lab 2. Deep sequence models.

## Class notes

One particular use case of RNN &rarr; video action recognition.

### Exercise 2.2
Histogram of the testing and training data distribution.
Apply the same data distribution to the encryptied and decrypted data. Beware that the number of characters in the two cases are not the same.

### Exercise 2.5
Compute different accuracies. One accuracy just for the words that are not corrupted (87.5 %), another one just for the words that are corrupted, and an overall accuracy. The first one is going to be used for checking that exercise 2.1 is correct (and that the model is robust to the corruptedness). The second one is useful for checking whether we are able to infer the missing word from the context.

Try bidirectional LSTMs and stacked LSTMs, and weight decay in the optimizer.

3 accuracies:
- 1 overall
- 1 accuracy only for the corrupted characters
- 1 accuracy only for the non-corrupted characters
