Padding is usually added whenever we want the output image to have the same size as the input (image segmentation, for example)


Useful website: https://poloclub.github.io/cnn-explainer/

Pytorch uses the `PIL` library to load images.

Numpy stores the dimensions of the images in a different way as the convolution wants them to be (numpy[H, W, C] vs torch[C, H, W]). Therefore, recall to permute the columns as needed.

```python
plt.imshow(convolution_grid.permute(1,2,0).detach().numpy()*10)
```

## Edge detection

> Edge detection is a binary classification problem

In an edge detection application, the goal is to obtain an edge mask, wich only has 1 channel. Edge detection is an unbalanced problem, because we have a lot more dark pixels than white pixels.

We multiply the loss function by 5 to penalize the network for not detectiong the edges.

```python
prob.detach().squeeze().cpu().numpy()
tensor.detatch() # Used to not compute the gradient
tensor.squeeze() # Remove the first dimension (Batch)
tensor.cpu() # Move the computation from GPU to CPU
numpy() # Convert the tensor to numpy
```

## Sizes of Convolutional Network for mnist dataset

The input is a batch of 128 images, 1 channel and the size of the images is 28*28. So the size is `[128, 1, 28, 28]`.

The first convolution produces an output of size `[128, 16, 28, 28]`, because we have padding=2 and 16 filters.
 
[128, 16, 14, 14] &rarr; maxpool
[128, 32, 14, 14] &rarr; convolution 2
[128, 32, 7, 7] &rarr; maxpool. This size is the one that we have to use for the last linear layer of the model (`7*7*32`).

After the flatten operation, we will have a [128, 32x7x7]


<h1><span style="color:red;">VERY SUPER MEGA HIPER IMPORTANT NEWS</span></h1>

We can store and load the results that we have obtained when training a model as follows

```python
torch.save(CNN.state_dict(), results_path+'/model.ckpt')
CNN.load_state_dict(torch.load(results_path+'/model.ckpt'))
```

# Suggestions

1. Inception / Resnet. Output passar-lo a maxpooling
2. Després utilitzar VGG, amb depthwise separable convolutions

Podem utilitzar més dades d'input, i transfer learning també.



