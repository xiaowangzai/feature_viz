# Feature Visualization
The purpose of this repo is to explore some of the differences in extracted features between residual networks and deep hybrid networks. 
The code for deep hybrid networks was adapted from [here](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/ScatteringTransform). 
The code for the residual networks was wrapped around tensorflow's [resnet implementation](https://github.com/tensorflow/models/tree/master/official/resnet).

# install
1. Clone this repo
1. Follow install instructions from [tensorflow object detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) and store in the resnet dirctory. 
1. Download the pretrained 50 layer residual network variables foune [here](https://arxiv.org/abs/1512.03385) and store it in the resnet directory. The pretraining was done on imagenet. 

To run the analysis launch the jupyter notebook and run the commands.
