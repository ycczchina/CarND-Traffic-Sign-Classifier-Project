# **Traffic Sign Recognition** 

## Writeup

### Zheng Chen

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./Writeup_images/Test_Hist.png "Visualization1"
[image2]: ./Writeup_images/Train_Hist.png "Visualization2"
[image3]: ./Writeup_images/Valid_Hist.png "Visualization3"
[image4]: ./Writeup_images/11.png "Traffic Sign 1"
[image5]: ./Writeup_images/13.png "Traffic Sign 2"
[image6]: ./Writeup_images/14.png "Traffic Sign 3"
[image7]: ./Writeup_images/17.png "Traffic Sign 4"
[image8]: ./Writeup_images/38.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?

  34799 images

* The size of the validation set is ?

  4410 images

* The size of test set is ?

  12630 images

* The shape of a traffic sign image is ?

  32*32 RGB

* The number of unique classes/labels in the data set is ?

  43 classes

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of data. 

![alt text][image1]

![alt text][image2]

![alt text][image3]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data because  this technique makes the data has mean zero and equal variance  and helps convolution.

```python
X_train = (X_train.astype(float) - 128.) / 128.
X_valid = (X_valid.astype(float) - 128.) / 128.
X_test = (X_test.astype(float) - 128.) / 128.
```


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5x3x6  | 1x1 stride, VALID padding, outputs 28x28x6 |
| Sigmoid	|												|
| Convolution 5x5x6x16	| 1x1 stride, VALID padding, outputs 24x24x16 |
| Sigmoid	|  |
| Convolution 5x5x16x36	| 1x1 stride, VALID padding, outputs 20x20x36 |
| Sigmoid	|  |
| Maxpooling 2x2	| 2x2 stride, VALID padding, outputs 10x10x36 |
| Flatten	| outputs 3600 |
| Fully connected	| outputs 120     |
| RELU	|         									|
| Dropout | 0.5 |
| Fully connected | outputs 84 |
| RELU |  |
| Dropout | 0.5 |
| Fully connected | outputs 43 |
| Softmax |  |
| Backprop |  |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `tf.train.AdamOptimizer()` with following parameters:

```python
EPOCHS = 100
BATCH_SIZE = 128
mu = 0
sigma = 0.1
rate = 0.005
dropout = 0.5
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used `evaluate()` to calculate the accuracy:

```python
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropout: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

My final model results were:
* validation set accuracy of 0.955
* test set accuracy of 0.950


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because lots of signs have a red triangle and black icon inside.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Yield   | Yield 					|
| Stop	| Stop							|
| No entry	| No entry		|
| Keep right	| Keep right   |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is higher than the accuracy on the test set. One reason is new images are not enough to cover all classes/evaluate the network.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 0.93). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|     .93     | Right-of-way at the next intersection |
| .06     				| Beware of ice/snow |
| .00					| Pedestrians	|
| .00	      			| General caution	|
| .00				    | Speed limit   (100km/h) |


For the second image, the model is relatively sure that this is a Yield sign (probability of 1.00). The top five soft max probabilities were

| Probability |      Prediction      |
| :---------: | :------------------: |
|    1.00     |        Yield         |
|     .00     | Speed limit (60km/h) |
|     .00     |       No entry       |
|     .00     |    Priority road     |
|     .00     | Speed limit (70km/h) |

For the third image, the model is relatively sure that this is a Stop sign (probability of 0.99). The top five soft max probabilities were

| Probability |      Prediction       |
| :---------: | :-------------------: |
|     .99     |         Stop          |
|     .00     | Speed limit (120km/h) |
|     .00     |     Priority road     |
|     .00     |       No entry        |
|     .00     |      Bumpy road       |

For the forth image, the model is relatively sure that this is a No entry sign (probability of 0.99). The top five soft max probabilities were

| Probability |  Prediction   |
| :---------: | :-----------: |
|     .99     |   No entry    |
|     .00     |     Stop      |
|     .00     |  Bumpy road   |
|     .00     | Priority road |
|     .00     |     Yield     |

For the fifth image, the model is relatively sure that this is a Keep right sign (probability of 1.00). The top five soft max probabilities were

| Probability |    Prediction     |
| :---------: | :---------------: |
|    1.00     |    Keep right     |
|     .00     |     No entry      |
|     .00     |  Turn left ahead  |
|     .00     |       Yield       |
|     .00     | End of no passing |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


