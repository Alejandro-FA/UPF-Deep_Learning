# Deep Learning Practice 1 report document

*Basic MLP implementation with Numpy/PyTorch using optimizers and autograd*

Group 8

Andreu Garcies (240618), Alejandro Fernández (242349), Marc Aguilar (242192)

## Exercise 1

First of all, after loading both train and test data we normalized it. Our first approach was doing it by dividing by the maximum value, but we were not getting correct results. Therefore, we decided to do it by substracting the mean and dividing by the standard deviation. 

Here we can observe both plots, before  and after normalizing the dataset.

![Figure 1](Results/fig1.png)
![Figure 2](Results/fig2.png)

### MLP class
In order to implement the multi layer perceptron (MLP) we used the class structure that we were given in the examples. We kept the sigmoid as the activation function but we changed the loss, since the MSE was not the appropiate function for a classification problem. Our choice was the Cross Entropy loss since it is used for solving classification problems, as in our case. We changed both the `loss` and added `lossPrime` where we computed the derivative, consequently slightly modifying the `backward` to use this new function. 

Our last modification to the MLP class was adding the parameter `update` to the `forward` function. We needed to do the plots where we compared the evolution of the loss between train and test data. 

### Train and test phase
We used the structure of the `train` function that we were given but we added another loss list in which we stored the values of the loss for the **test data**.

We decided to keep both the number of epochs to `epoch=10000` and the learning rate `lr=0.01` for training. On the other hand, we set the number of **hidden neurons** up to **20** We can observe that the accuracy for the training dataset is very high **(97%)** as we initially expected. It only misclassifies 3 datapoints. 

![Figure 3](Results/fig3.png)

Then, regarding the evolution of the loss we can observe how the test one is higher than the train one and there is not a significant sign of overfitting. From iteration 80 the test loss seems to grow a little bit so we could get a little bit of overfitting but not very significant.

![Figure 4](Results/fig4.png)

Lastly, the classification of the test dataset gives us very good results too. As we can observe, it only misclassifies 3 datapoints. We reach an accuracy of **(88%)**, which is quite high. 

![Figure 4](Results/fig5.png)


## Exercise 2

### Implementation details

In this section we will focus on explaining what difficulties we encountered to justify our implementation choices. The foremost detail to remark is what **version of momentum we have decided to implement**. At first we settled for the version that multiplies the gradient by $1 - \beta$:

$$
\displaylines{
V_t = \beta V_{t-1} + (1-\beta)\nabla_wL(W, X, y) \\
W = W - \alpha V_t
}
$$

But after comparing the results with the standard Stochastic Gradient Descent we saw that the results were not as expected. After some thinking, we realised that we had forgotten about scaling the learning rate, otherwise its value was not comparable between exercises. To avoid this type of confusions, we thought that it was better to implement the second alternative, which uses a comparable learning rate:
$$
\displaylines{
V_t = \beta V_{t-1} + \alpha\nabla_wL(W, X, y) \\
W = W - V_t
}
$$
Other details that should be noted is that we have used a different moving average for each layer, because the weights update in each layer can be completely different. Furthermore, all $V_t$ are initialised to $0$ at the beginning. With this implementation, the result that we obtained was the following one:

<img src="Results/fig6.png" alt="fig6" style="zoom: 33%;" />

It is clear that the addition of momentum has a positive effect, both in time required to reach the minimum and the minimum error that we get. It is important to note that we have used a fixed random seed to ensure that the results are easily comparable.

### Trying multiple values of $\beta$

When trying multiple values of $\beta$ we have also ensured to have the same random numbers for each test, to reduce as many distortion of the results as possible.

<img src="Results/fig7.png" alt="fig7" style="zoom:33%;" />

The result obtained is pretty straightforward to interpret. The higher the value of $\beta$ (up to $0.9$), the better the result. This should not come as a big surprise, as a small value of $\beta$ implies that we significantly reduce the effect of the momentum. This results is in line with the rule of thumb of using a beta of $0.9$ as a good starting point.

## Exercise 3

Once we have implemented the MLP and tested that it works as expected, we can start changing the hyperparameters to understand how the network behaves for this particular problem of binary classification. In this section, we will show the results that we have obtained for the different number of hidden neurons and the different values of the learning rate, and we will discuss  what we believe that is the best hyper parameters choice.

### Trying with different number of hidden neurons

The following image shows the evolution of the loss function with respect to the number of iterations for the testing dataset.

<img src="Results/fig9.png" alt="fig9" style="zoom:35%;" />

The more hidden neurons the network has, the less number of iterations are required for the error to decrease. That is, the less number of iterations are required for the network to be able to properly classify the testing dataset. By observing the graph, we can see a pattern regardless of the number of neurons the network has: after reaching the minimum loss value, in all cases overfitting starts to be visible from a graphical point of view. However, we need to be aware of the very small range of values in which the error oscilates as iterations increase $[\approx0.146, \approx 0.149]$. Therefore, we can consider this overfitting to be negligible.

On the first place, our attention has been caught by the network with $7$ hidden nodes, as it is the one that achieves the smallest error value with a small number of parameters due to the reduced number of neurons that it has. The higher the number of neurons the network has, the less iterations are required to reach to the point of minumum error. Nevertheless, what we believe that is the most important, is the amount of time that it takes for the neuron to reach that point. Therefore, to have a better understanding of how the different networks perform, we have decided to measure the time that it takes for them to reach the minimum error. The following table shows the results:

| Number of hidden neurons | Minimum Error value | Time (ms) |
| ------------------------ | ------------------- | --------- |
| 3                        | 0.151               | 0.23      |
| **7**                    | **0.146**           | **0.25**  |
| 12                       | 0.149               | 0.24      |
| 15                       | 0.149               | 0.75      |
| 18                       | 0.147               | 0.25      |
| 20                       | 0.148               | 0.27      |
| 50                       | 0.147               | 0.29      |
| **100**                  | **0.148**           | **0.30**  |

> These time results have been achieved with a 2015 iMac, 4 cores 4 GHz Intel Core i7 CPU and 32 GB 1867 MHz DDR3 memory. Resluts may vary from one machine to another.

As we can observe, time varies very little between the different networks.

Having said all of this, we have decided that the best network for this particular problem is the one with just 7 hidden neurons. It is the one that achieves the smallest error and it does so in the same time (approximately) as larger networks reach their own minumum error value (even though it takes more iterations).

### Trying with different learning rate values



<img src="Results/fig10.png" alt="fig10" style="zoom:35%;" />
