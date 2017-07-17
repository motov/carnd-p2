#**Traffic Sign Recognition**

##Alex Liu

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_img/img1.png "Visualization"
[image2]: ./write_up_img/img1_gray.png "Grayscaling"
[image3]: ./write_up_img/img2.png "valid img2"
[image4]: ./write_up_img/img3.png "valid img2"
[image5]: ./examples/random_noise.jpg "Random Noise"
[image6]: ./write_up_img/8.jpg "speed limit 60"
[image7]: ./write_up_img/1.jpg "stop sign"
[image8]: ./write_up_img/2.jpg "No stopping"
[image9]: ./write_up_img/4.jpg "pedestrians"
[image10]: ./write_up_img/6.jpg "Speed limit 30"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43


Here are some images from training & validation data sets.


![alt text][image1]
![alt text][image3]
![alt text][image4]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it is easier to train grayscale than 3 layer RGB images. I tried both open cv functions and direct calculation from RGB space to grayscale. I found that the results are similar that they both get similar training loss with same setup.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1]
![alt text][image2]

As a last step, I normalized the image data because it is required for the network to train the data since they are all numerical operations. Without normalization, loss is huge.




My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, RELU, outputs 32x32x32 	|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
|Dropout | keep_prob = 0.6|
| Convolution 3x3	    | 1x1 stride, valid padding, RELU, outputs 16x16x32      				|
| Max pooling         | 2x2 stride,  outputs 8x8x32|
|Dropout | keep_prob = 0.6|
|Convolution 3x3|1x1 stride, valid padding, RELU, outputs 16x16x64 |
| Max pooling         | 2x2 stride,  outputs 4x4x64|
|Dropout | keep_prob = 0.6|
|Convolution 3x3|1x1 stride, valid padding, outputs 4x4x64 |
|Dropout | keep_prob = 0.6|
| Fully connected		| 1024 --> 256.        									|
|Dropout | keep_prob = 0.6|
| Fully connected		| 256 --> 128.        									|
|Dropout | keep_prob = 0.6|
| Fully connected		| 128 --> 43.        									|

I spent huge amount of time training the model. The reason is that I made a big mistake when training the model. When training, I add dropout after each layer, but during validation, I didn't delete dropout layer.

This makes my validation accuracy extremely difficult to be higher than 0.93. I tried a lot of different structures with huge combinations of hyperparameters but validation accuracy can't go up.

After I fixed this problem, the model is easy to meet 0.93 validation accuracy.

My model is different than LeNet model. I found that add more convolution layers is better. Also making the depth of convolution layer higher is helpful. That's why I added 2 more convolution layers. Also, adding dropout helps improving the validation accuracy.

My final model results were:
* training set accuracy of 0.972
* validation set accuracy of 0.935
* test set accuracy of 0.916




###Test a Model on New Images


Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| speed limit 60		| speed limit 60     							|
| Stop Sign      		| Stop sign   									|
| No stopping     			| keep right										|
| pedestrians					| traffic signal											|
| speed limit 30      		| speed limit 30				 				|



The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This accuracy is lower than the test set. I think it is because we have only 5 samples here. If I have a lot more images, the accuracy will go up.

Also I find that the second image actually doesn't exist in the training and validation sets. That's why it is not recgonized. Thus the total accuracy should be 3/4 = 75%

#### Below are the top 5 probabilities for each web image.

For the first image, the model is relatively sure that this is a speed limit 60 (probability of 0.9), and the image does contain 60 speed limit. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 90%         			| speed limit 60   									|
| 3.5%     				| speed limit 80 										|
| 3.2%					| speed limit 50											|
| 2.4%	      			| speed limit 30					 				|
| 0.5%				    | speed limit 20      							|


For the second image, the model is extremely certain that it is a stop sign which is correct.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 99.9%         			| stop sign   									|
| 0.00006%     				| speed limit 30 										|

For the third image, since it is not contained in the training set. It will provide a wrong answer. But it is worth seeing what happens if the testing image is not included in the training and validation sets. The label should be no stopping which doesn't exist in the testing labels. As is seen below, the prediction is not randomly choosen from the labels with similar probabilities. In fact, it is giving a very high Probability that the sign is turn right sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 99.85%         			| turn right   									|
| 0.01%     				| turn left 										|


For the fourth image, the model is not very certain about which traffic sign it is. The label should be pedestrians however it predicted traffic signal sign with probability of 54%. And the model predicted it as pedestrians with probability of 30%

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 54%        			| traffic signal   									|
| 29%     				| pedestrians 										|
| 12%     				| General caution 										|
| 1.8%     				| Children crossing 										|
| 1.1%     				| Road narrows on the right									|

For the fifth image, the model is quite certain that it is a speed limit 30 sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 83%        			| speed limit 30 									|
| 15%     				| speed limit 20 										|
| 1.7%     				| speed limit 50										|
| 0.6%     				| speed limit 80										|

Overall, I think the model has shown that it has similar accuracy to the test sets. Once we have more web images, the accuracy will go up. There are a lot more to boost the accuracy. Here are the list I can think of:
1. add more epochs, I tried with 100 epochs and it did improved validation accuracy.
2. use a more complex model.
3. data augment.
4. L2 regularization
5. fine tuning for specific traffic signs that have low accuracy.
