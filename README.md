## 1. Introduction

## 2. Data and EDA

## 3. Model
### 3.1 Model 1 - Kaggle Competition model
Kaggle hosted the competition for facial emotion recognition in 2016. In this competition, 7 classes of facial emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral) are classified using CNN model. For our project, we are interested in identifying students’ confusion and distraction emotion. Hence we add two more classes (confused and distraction) to the input with custom labelled data.  The model architecture is shown below with 4 convolution layers and 2 fully connected dense layers. Batch normalization and dropout are used to future improve model performance. The model loss function is categorical cross-entropy and the model optimizer is Adam. Model loss and accuracy were plotted for at different epochs. The train accuracy is 0.75 while test accuracy is 0.62. Based on the confusion matrix (normalized) results, one can see that the model has poor performance. The model generalized the facial emotions more to “happy”, “neutral” and “sad”. We also conduct real-time tests with jetson TX2 webcam and the model is not able to correctly classify facial emotions. One possible reason for low model performance is the resolution of the image and the color of the image (grey). Lower resolution and grey color mean that there are less features to extract and generalize the model prediction. Therefore we tried to improve the model performance by using customized RGB images with high resolution (224x224).
### 3.2 Model 2 - littleVGG
Another model we have tried is called littleVGG and the architecture is shown below. It has 6 convolutional layers. For this model, we incorporate data augmentations, such as rescale 1/255, rotation rage = 30, shear range = 30, zoom range =30, as well horizontal flip. THe loss function is categorical cross entropy, and the optimizer is Adam with learning rate of 0.0001 and decay of 1e-6. 

Experiments have been done with model trained for 9 classes, 7 classes and 5 classes, and we obtained roughly similar results. The normalized confusion matrix results of 7 classes are shown below. Among 7 classes, the customized classes are confusion and distraction. The validation results showed well performed predictions for all classes but distraction.
### 3.3 Model 3 - Transfer Learning with Resnet50
Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. In the domain of deep learning, transfer learning usually involves using a pre-trained model, which has been previously trained on a dataset and contains the weights and biases that represent the features of the dataset, to “learn” something else and by doing so, the training time will be significantly reduced without compromising the model performance. 

Here we used a pre-trained model Resnet50 to conduct transfer learning for facial expression classification. Resnet50 was built on top of VGG19 and has more layers and better performance. For Resnet50, researchers reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. With an ensemble of these residual nets, Resnet50 achieved an error rate of as low as 3.57% on the ImageNet test set, and championed in the ILSVRC 2015 classification task. 
 
The architecture of the transfer learning model consists of Resnet50 and one additional dense layer. The model used sgd as an optimizer with learning rate of 0.1, decay = 1e-6, and momentum = 0.9. The loss function is categorical cross entropy. The model has 10 epochs and the train/test accuracy reached almost 100% after 3 epochs, indicating very fast convergence and high level of performance. The model weights were saved in .h5 format and loaded on the edge device (Jetson TX2) for inference. 

## 4. Pipeline
## 5. End Product
## 6. Model Performance
## 7. Future Steps
