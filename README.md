**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_for_report/samples.png "Samples"
[image2]: ./images_for_report/class_distribution.png "Class Distribution"
[image3]: ./images_for_report/visualised_images.png "Visualised images"
[image4]: ./images_for_report/adaptive_histogram.png "Adaptive histogram applied to images"
[image5]: ./images_for_report/new_examples.png "New examples and probabilities"

### Data Set Summary & Exploration

I used the matplotlib library to visualise samples of the traffic signs data set.

![alt text][image1]

Some summary statistics calculated using the numpy library are as follows:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

Here is a bar chart showing how the distribution of classes by training, validation and testing set. We see that each class is not represented equally. However, the training, validation and testing sets have almost the same distribution of classes. 

To account for the varied distribution of classes, when training our model, we could sample each class equally.

![alt text][image2]

### Design and Test a Model Architecture

#### Preprocesssing

It was noted that the images visualised showed very similar orientations and scale and mainly differed in lighting conditions. 

![alt text][image3]

Thus to make the model more robust to differences in lightning, adaptive histogram normalisation was performed on each image for preprocessing.

![alt text][image4]

I tested data augmentation with Keras (rotation and scale), but the model didn't seem to perform significantly better. This was likely due to the data having very similar rotations and scale.

Greyscale was investigated, however it didn't show significant performance gains either. It was hypothesised that colour would be a key distinguishing feature since some traffic signs show strong yellow, red and blue colours. 

#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Dropout				| keep_prob										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64	|
| RELU					|												|
| Dropout				| keep_prob										|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Flatten		      	| outputs 5x5x64 = 1600							|
| Fully connected		| outputs 960  									|
| RELU					|												|
| Dropout				| keep_prob										|
| Fully connected		| outputs 336  									|
| RELU					|												|
| Dropout				| keep_prob										|
| Fully connected		| outputs classes								|
| Softmax				| 	        									|

Dropout was used to reduce overfitting. A keep_prob of 0.5 was used for the dropout layers in training to perform regularisation and 1.0 was used in the final model to make use of the full model. 

It was thought that the LeNet architecture was underfitting the model, so more neurons were added as extra depth dimensions in the architecture. This improved the performance of the model. 

#### Model Training

The model was trained on a model using the architecture described above. An Adam Optimiser was used with a learning rate of 0.001. 

50 epochs were used with a batch size of 256. It was thought that the model had converged sufficiently after the 50th epoch. A batch size of 256 was used as it seemed a good compromise between being large enough to avoid overfitting to the data, but small enough that the model is able to learn well enough. 

#### Solution Approach

To achieve a validation set accuracy of at least 0.93, I first tried to maximise validation accuracy whilst being cautious of overfitting. 

I used the LeNet architecture as a baseline with hyperparameters taken from the tutorial examples as these performed reasonably well with the tutorial examples. 

Each iteration would consist of minor changes so that any increases / decreases in performance could be easily attributed to particular changes. I used a strategy inspired by genetic algorithms. After each iteration of changing hyperaparameters and/or model architecture, I would review the validation accuracy and decide whether this was an improvement. If an improvement was made, I would keep the change and perhaps make a more significant change. 

The main changes made between iterations consisted of tuning the batch size and epochs hyperparameters as well as changes to the model architecture including adding dropout layers, the size of the convolution layers and the number of neurons at each layer. 

Once I had achieved a significant improvement in validation accuracy, I would test the model on the test set. This was done sparingly to avoid overfitting to the test set. 

My final model results were:
* training set accuracy of 0.984
* validation set accuracy of 0.984
* test set accuracy of 0.972

###Test a Model on New Images

#### Acquiring new images

Here are seven German traffic signs that I found on the web as well as the associated top 5 softmax probabilities for each example.

![alt text][image5]

The model correctly predicts 5 out of 7 images. Where it performs well it predicts the correct classification with high confidence. The model fails to predict "End of speed limit 80 km / h" and "Road work". All images were similar to the training data in that they had a tight bounding box around them and were oriented well. 

It's possible the model failed to classify "End of speed limit 80 km / h" correctly due to it being too side on (where most of the training data is facing front on). Further analysis needs to be performed on "Road work" misclassification as it is not obvious why this was the case. It is possible the model has low recall on "Road work" signs and so its confidence that it was a "Road work" sign is quite low. The "Road work" sign looks quite similar to the "Road narrows on the right" sign so that would explain why that classification ranks second. However, it is unclear why the "Bicycles crossing" classification was made.

#### Performing on New Images

The accuracy of the model on the new images is 5 / 7 = 71%, significantly lower than that of the test set. It is likely this is lower because the new images are greatly different from those of the training, test and validation set. Since the test set comes from the same database of images, there are likely to be similarities in how these images were collected (perhaps same set of cameras and region of interest finding). This contrasts to my unstandardised images found on the internet. 


