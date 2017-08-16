# DCGAN
PyTorch implementation of Deep Convolutional Generative Adversarial Networks (DCGAN)

## Generating MNIST dataset
### Network architecture
* Input data
    * MNIST data is resized to 64x64 size images
* Generator
    * hidden layers: Four 4x4 strided convolutional layers (1024, 512, 256, and 128 kernels, respectively), ReLU activation
    * output layer: 4x4 strided convolutional layer (1024 nodes = 64x64 size image), Tanh activation
    * Batch normalization is used except for output layer

* Discriminator
    * hidden layers: Four 4x4 convolutional layers (128, 256, 512, and 1024 kernels, respectively), Leaky ReLU activation
    * output layer: 4x4 convolutional layer (1 node), Sigmoid activation
    * Batch normalization is used except for 1st hidden layer & output layer
    
### Results
* For learning rate = 0.0002 (Adam optimizer), batch size = 128, # of epochs = 30:
<table align='center'>
<tr align='center'>
<td> GAN losses</td>
<td> Generated images</td>
</tr>
<tr>
<td><img src = 'MNIST_DCGAN_results/MNIST_DCGAN_losses_epochs_30.gif'>
<td><img src = 'MNIST_DCGAN_results/MNIST_DCGAN_epochs_30.gif'>
</tr>
</table>

### References
1. http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
2. http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
3. https://github.com/hwalsuklee/tensorflow-GAN-1d-gaussian-ex


