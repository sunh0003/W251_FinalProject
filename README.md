## 1. Introduction
Distance learning or e-learning has been trending in these years because of its advantage in providing education opportunities at reasonable costs to those who cannot be on campus all the time. Distance learning can be uptaken in multiple ways, including computer-based learning (e.g. software assisted learning) and internet-based learning (e.g. pre-recorded online lecture). Among all the options available, real-time interactive virtual classroom is closest to traditional face-to-face teaching and learning and is also most technologically intense. 

By the end of 2019, virtual classrooms have been mostly used to deliver higher education courses, but since the breakout of COVID-19 in early 2020, virtual classrooms started to take over the K-12 education as well when parents, children and teachers hastily tried to figure out a safe learning environment. According to a AACSB Quick-Take Survey, globally 51 percent of respondents had converted face-to-face courses to online or virtual formats, with some regional differences. 

There are many commercial softwares to facilitate virtual classrooms, with Zoom, the video chatting application as the leading player. Compared to December 2019, the daily users in Zoom meetings increased by 20 folds in March 2020 and has kept increasing as of today.

![Alt text](images/intro.png?raw=true "Fig1-1, introduction")

Despite all the benefits virtual classrooms can bring in, there is still concern that the productivity may not be comparable to that of the face-to-face classrooms. Generally, people are not quite sure about the effectiveness of course delivery and the engagement of students. Here we would like to introduce Virtual TA, a prototype of deep learning based facial expression recognition application, which monitors students’ instant facial expressions and helps the teacher to address confusion or misunderstanding immediately and to maintain the students’ engagements and motivations. 

In order to get an idea of what the potential users of Virtual TA want to get out from this application, we did a mini market survey. We sent out a survey with 6 questions to the instructors who teach virtual classes. According to the responses, the major pain point for instructors in teaching a virtual class is how to figure out whether the students understood the concept and how to get them to engage and participate. About 80% of instructors would like to use Virtual TA if it’s available and also 80% think a summary of facial expressions will help them on the teaching work. Specifically, 60% think the most helpful expression is the confused and they would like to see statistics on level of interest and engagement, overall mood or sentiments during a lecture and across sections. 

The model used by Virtual TA for facial recognition and expression classification is deep neural network, which is trained on cloud and deployed on edge such as the laptop or the Nvidia Jetson TX2. Virtual TA has two aspects of functions: 1. It can capture faces from a virtual classroom using a web camera, and then classify the expression for real time; 2. It can process a recorded video, classify the expressions and output statistics of interest, for example the overall confused level during some period of a lecture. The success of Virtual TA largely depends on the model performance. We assume that with more usage, more data can be obtained and used to further train the deep neural network, making it more accurate and more generalized. 

## 2. Data and EDA
### 2.1 Kaggle competition dataset
Initial dataset was downloaded from Kaggle competition “Facial expression recognition with deep learning”. The dataset consists of 36000 images for 7 classes, which are Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The images are 48x48 pixels and grey in color. However, the problem of using this dataset is first, it doesn’t have any asain face and second, it doesn’t have the class of most interest. We tried to generate the missing part of data and combine it with the Kaggle dataset but didn’t see significant improvements on model performance. We hypothesized that higher resolution may be needed so we decided not to use this dataset for model training and validation.

![Test Image 2](images/kaggle_dataset.png?raw=true width=400)
### 2.2 Cohn-Kanade (CK) AU-Coded Facial Expression dataset
The Cohn-Kanade (CK) AU-coded Facial Expression dataset includes 2000 images sequences from over 200 subjects (university students). This dataset was analysed by Cohn, Zlochower, LIen, & Kanade (1999) and by Lien, Kanade, Cohn, & Li (2000). These papers can be downloaded from http://www.cs.cmu.edu/~face. These images are 480x480 pixels with grey scale. The image sequences from neutral to target motion and the target emotion is the last frame (as shown below). The final frame of each image sequence was coded using FACS (Facial Action Coding System) which describes subjects’s expression in terms of action units (AUs). An Excel spreadsheet containing these FACS codes is available for our analysis. We did a round of investigation of these images and did not use this set of images in our final model to the complexity of the FACS coding system as well as the image sequence. The image sequence makes it harder for us to tell which image we need to use in our training model. 

![Test Image 3](images/CK_dataset.png?raw=true width=400)
### 2.3 Custom dataset
For our final model, we eventually chose to use our own custom dataset. This dataset contains more than 4000 images from 6 subjects and the sample images are shown below. The images are 224x224 pixels with color scale (RGB). This dataset contains 3 classes which are confused, happy, and surprised. The image was captured by Jetson TX2, face is cropped to 224x224 using the open-cv face detection using Haar Cascades similar to HW03. After faces are cropped out and categorized into different classes, images are uploaded to VM in ibm cloud for training/testing the model. 

![Test Image 4](images/us_dataset.png?raw=true width=400)

## 3. Model
### 3.1 Model 1 - Kaggle Competition model
Kaggle hosted the competition for facial emotion recognition in 2016. In this competition, 7 classes of facial emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral) are classified using CNN model. For our project, we are interested in identifying students’ confusion and distraction emotion. Hence we add two more classes (confused and distraction) to the input with custom labelled data.  The model architecture is shown below with 4 convolution layers and 2 fully connected dense layers. Batch normalization and dropout are used to future improve model performance. The model loss function is categorical cross-entropy and the model optimizer is Adam. Model loss and accuracy were plotted for at different epochs. The train accuracy is 0.75 while test accuracy is 0.62. Based on the confusion matrix (normalized) results, one can see that the model has poor performance. The model generalized the facial emotions more to “happy”, “neutral” and “sad”. We also conduct real-time tests with jetson TX2 webcam and the model is not able to correctly classify facial emotions. One possible reason for low model performance is the resolution of the image and the color of the image (grey). Lower resolution and grey color mean that there are less features to extract and generalize the model prediction. Therefore we tried to improve the model performance by using customized RGB images with high resolution (224x224).

![Test Image 5](images/kagglemodel.png?raw=true)

### 3.2 Model 2 - littleVGG
Another model we have tried is called littleVGG and the architecture is shown below. It has 6 convolutional layers. For this model, we incorporate data augmentations, such as rescale 1/255, rotation rage = 30, shear range = 30, zoom range =30, as well horizontal flip. THe loss function is categorical cross entropy, and the optimizer is Adam with learning rate of 0.0001 and decay of 1e-6. 

Experiments have been done with model trained for 9 classes, 7 classes and 5 classes, and we obtained roughly similar results. The normalized confusion matrix results of 7 classes are shown below. Among 7 classes, the customized classes are confusion and distraction. The validation results showed well performed predictions for all classes but distraction.
### 3.3 Model 3 - Transfer Learning with Resnet50
Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. In the domain of deep learning, transfer learning usually involves using a pre-trained model, which has been previously trained on a dataset and contains the weights and biases that represent the features of the dataset, to “learn” something else and by doing so, the training time will be significantly reduced without compromising the model performance. 

Here we used a pre-trained model Resnet50 to conduct transfer learning for facial expression classification. Resnet50 was built on top of VGG19 and has more layers and better performance. For Resnet50, researchers reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. With an ensemble of these residual nets, Resnet50 achieved an error rate of as low as 3.57% on the ImageNet test set, and championed in the ILSVRC 2015 classification task. 
 
The architecture of the transfer learning model consists of Resnet50 and one additional dense layer. The model used sgd as an optimizer with learning rate of 0.1, decay = 1e-6, and momentum = 0.9. The loss function is categorical cross entropy. The model has 10 epochs and the train/test accuracy reached almost 100% after 3 epochs, indicating very fast convergence and high level of performance. The model weights were saved in .h5 format and loaded on the edge device (Jetson TX2) for inference. 

![Test Image 6](images/resnet50.png?raw=true)

## 4. Pipeline

## 5. End Product
## 6. Model Performance
## 7. Future Steps
