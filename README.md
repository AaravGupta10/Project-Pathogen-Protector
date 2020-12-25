# Project Pathogen Protector

## Overview
We felt the need to make such a project because of the ongoing pandemic scenario (as of 2020) which has affected many people in various stages of life. It has also affected many businesses and led to the closure of many institutions like schools, colleges, offices, etc. Now, since the world opening and various activities have started again, we must take utmost precaution at traveling outside and in public places. The best way one can protect oneself and the community from the SARS-Cov-2 virus is to wear masks which protect us by not letting viruses (and other pathogens) enter into our body and others by not letting our pathogens enter into them. However, some people do not realize the importance of wearing a mask and put their as well as others’ lives at risk. It will be very essential to prevent this from happening when schools reopen.
Thus, we have come up with a brilliant solution. We have created a program capable of detecting if someone is wearing a mask or not and recording the names of those who are not wearing masks (given that at least one photo of the child/student is in our database). The names of those who have not worn masks are recorded in a .csv file and this information is then conveyed to the respective class teachers through email. This has been possible because of technologies like Artificial intelligence, Machine learning, and Convolutional Neural Networks (Deep Learning). We have majorly used the programming language, python, to accomplish this along with several frameworks and libraries including Keras, Numpy, OpenCV. 

## Process
Our whole process of analyzing, developing, creating, and evaluating is based on the design cycle. 

### Data Processing
The first step we took was to find a sufficient dataset of pictures with and without masks which we needed to train a convolutional neural network (a neural network best for identifying specific objects like masks). We found a dataset that consisted of 690 images with masks and 686 without. We then converted the images to grayscale (100×100) using the OpenCV library. This makes it easy to train the neural network. Since the neural network requires a 4D array, we then converted the data into a 4D array using the Numpy library. 
For facial recognition, we used face-recognition library along with C++ dependencies like CMake and DLib. For the face recognitional program, only a single photo of the individual is required to train the model as it is a very high-level model.

### Training
We created the neural network architecture using the Keras library in python. The first and second layers of the neural network had 200 3×3 kernels and 100 3×3 kernels followed by a ReLU and MaxPooling2D layer. Then there was a flatten layer to stack the output convolutions from the second convolution layer and a dropout layer to reduce the overfitting. Then there was a dense layer of 50 neurons followed by the final layer consisting of 2 neurons (for with and without mask). Next, we went through around 20 epochs through the dataset using the sklearn and created several models. Using the matplotlib library in python we created several graphs to analyze and choose the best model. 
The face-recognition library simplifies the training process a lot for the facial recognition part as most of the encoding and training work is done by the library automatically.

### Testing and Implementing
The final step was to utilize the trained model to detect faces with and without masks. We used a pre-trained .xml document (haar cascade classifier) to identify the person's face. We used the cascade classifier in grayscale as it is easier to work with. The program first runs the cascade classifier then it runs the trained model. The output of the program is displayed in a new window and shows a live stream of the program.
We built the program logic in such a way that the face recognition program runs only when a person is not wearing a mask. The name and time of the person are then recorded and stored in a .csv file. Finally, we created an email account (noreply.pathogenprotector@gmail.com) from which we can send the .csv file to the respective teachers (on outlook) and implemented that in our code using some built-in python SMTP libraries. The email is sent when the recognition and mask detection part is completed (by pressing the escape key).

### Dataset used:
https://github.com/prajnasb/observations/tree/master/experiements/data
