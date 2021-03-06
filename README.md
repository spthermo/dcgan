# DCGAN
This is an abstract implementation of a DCGAN architecture proposed in [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). The model is developed in pytorch and trained on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) face dataset.

## Prerequisites
The architecture has been implemented using the following:
- Python 3.5
- Scipy
- Torchvision
- Tensorflow 1.7.0
- Tensorboard

Tensorflow and Tensorboard are used for visualization and monitoring purposes, thus they are not mandatory.

## Running the code
To start training the dcgan model use:
```
python main.py --dataPath /path/to/celebA
```

The ```logger.py``` file is used to create and update the model's instance for Tensorboard. To monitor the training process use:
```
tensorboard --logdir='./logs' --port 6006
```
and use your browser to access the localhost at the specified port.

## Examples
Some indicative examples of face images generated by the dcgan generator are:

<img src="https://github.com/spthermo/dcgan/blob/master/examples/1.png" width="100"> <img src="https://github.com/spthermo/dcgan/blob/master/examples/2.png" width="100"> <img src="https://github.com/spthermo/dcgan/blob/master/examples/3.png" width="100"> <img src="https://github.com/spthermo/dcgan/blob/master/examples/4.png" width="100"> <img src="https://github.com/spthermo/dcgan/blob/master/examples/5.png" width="100">

## Acknowledgement
This work is based on the PyTorch examples. The Tensorboard support is provided from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard).



