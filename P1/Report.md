# Deep Learning Practice 1 report document

*Basic MLP implementation with Numpy/PyTorch using optimizers and autograd*

Group 8

Andreu Garcies (240618), Alejandro Fernández (242349), Marc Aguilar (242192)

## Exercise 1

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
