# Class-Activation-Map-CAM-and-Grad-CAM

## Problem Definition
For many years, Convolutional Neural Networks were treated as black boxes with inputs and outputs, but we didn't know what the network was learning. Class Activation Map (CAM) aids in understanding what features the network is learning and which areas are prioritized in order to distinguish classes from one another.

In this work, I have used CAM and Grad-CAM in a pokemon classification problem setting to learn more about the internal workings of a deep CNN. This problem statement will allow to fine-tune pre-trained networks on a new classification task, followed by an analysis of the newly trained network to see what features it is learning.

<img src="https://github.com/shivaniNK8/Class-Activation-Map-CAM-and-Grad-CAM/blob/main/images/Screenshot%202023-03-30%20at%201.51.20%20PM.png" width=50% height=50%>

<img src="https://github.com/shivaniNK8/Class-Activation-Map-CAM-and-Grad-CAM/blob/main/images/cam.png" width=50% height=50%>

## Dataset
Dataset Link: https://www.kaggle.com/datasets/thedagger/pokemon-generation-one

The dataset consists of 151 pokemons, with 60 images each per pokemon. Out of these 151 pokemons, 7 classes were selected for analysis.

## Result Images:
### Combined Results of AlexNet, VGG 16 CAM and VGG19 Grad-CAM
<img src="https://github.com/shivaniNK8/Class-Activation-Map-CAM-and-Grad-CAM/blob/main/images/combine_cam.png" width=50% height=50%>

### Alexnet CAM Results
<img src="https://github.com/shivaniNK8/Class-Activation-Map-CAM-and-Grad-CAM/blob/main/images/alex_cam.png" width=50% height=50%>

### VGG16 CAM Results
<img src="https://github.com/shivaniNK8/Class-Activation-Map-CAM-and-Grad-CAM/blob/main/images/vgg_cam.png" width=50% height=50%>

