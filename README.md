# ML-MNIST

Machine learning project for the MNIST dataset.

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains handwritten digits with a training set of 60000
images and a test set of 10000 images.

## Results

Model                                   |Test Accuracy
----------------------------------------|-------------
Dense 10 hidden layers with 100 units   |97.8%
CNN with 1 conv. layers                 |98.2%
CNN with 2 conv. layers                 |98.7%
CNN with 2 conv. layers and dropout     |98.7%
CNN with 2 conv. layers, dropout and batchnorm. |98.9%
fastai Resnet18                         |99.5%