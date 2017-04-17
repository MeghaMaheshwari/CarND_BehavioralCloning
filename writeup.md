## <b>Udacity Self-Driving Nanodegree 
<hr/>


## Behavioral Cloning for autonomous cars
======================================================================

### Build a network that uses behavioral cloning to drive cars in autonomous mode

> **The Steps in the Project are as following :**

> - Collect the training data using the simulator in training mode
> - Define a network for training the data
> - Validate the model architecture on the validation set. 
> - Test the output of the model architecture on the simulator in autonomous mode.
> - Record the video of the car in autonomous mode.
> - Summarize the results with a written report.


#### <b>Collecting the training data</b>

The data set for this project is collected by running the simulator in training mode. I have gathered data for two tracks to have sufficient information to
train the network. Also since the first track is biased towards left side mostly, I used data from the second track as well so that we have information on
driving towards the right side as well as on the bridge.
   The simulator generates the following training data

> - Colored images of size 320x160 with 3 channels as its a colored image.
> - A csv file containing the following information

  <table style="width:100%; align:center;">
  <tr>
    <th>Image from the center camera of the car</th>
    <th>Image from the left camera of the car</th>
	<th>Image from the right camera of the car</th>
	<th>Steering Angle</th>
	<th>Throttle</th>
	<th>Break</th>
	<th>Speed</th>
  </tr>
  <tr>
    <td>center_2017_04_14_10_51_31_160.jpg</td>
    <td>left_2017_04_14_10_51_31_160.jpg</td>
    <td>right_2017_04_14_10_51_31_225.jpg</td>
    <td>0</td>
	<td>0</td>
	<td>0</td>
	<td>0.000003835525</td>
  </tr>
  </table>
  
  Some of the captured training images can be seen below:
  
	<img src="example/center_2017_04_14_10_51_31_160.jpg" width="150" style="margin-right: 50px;"> <img src="example/left_2017_04_14_10_51_31_160.jpg" width="150" style="margin-right: 50px;">  <img src="example/right_2017_04_14_10_51_31_225.jpg" width="150" style="margin-right: 50px;">  


#### <b>Defining the network</b>
Keras has been used to create the network with tensor flow as backend.
The images are loaded by taking their location from the csv file. The network is being trained only on the steering angle in this project and
hence only the third row is being considered for the labels from the csv file.
###### Data Augmentation
 Data augmentation is needed here to provide the network with additional data for training. Since we know that most images could either be left biased
 or right biased, we flip the images and augment the data set so that the network gets a better understanding of the steering direction from each side.

###### Preprocessing the data
The following steps have been used for preprocessing.
> - Since the upper part of the image consists of mostly the hills, sky and trees, the image is cropped so that the network can focus only on relevant information.
> - The images are normalized to maintain numeric stability.

###### The network

 A convolutional network has been used in this case with 6 filters and a kernel size of 5 x 5. By experimenting it was found that
 the loss on training data was much better than on the validation set which seemed to be due to overfitting. Hence, after experimentation, it was seen
 that a droput of .3 gave a much better accuracy on the validation set. Adam optimzer has been used to minimise the mean square error loss.
 The training data has been split into training and validation sets as 80% and 20%. This value was also obtained after experimentation.
 Initially the number of epochs that I chose was 10, but than I saw that 7 epochs gave a reasonable accuracy on the training and validation sets.
 
 The entire network can be visualized by the following table:

<table style="width:100%; align:center;">
  <tr>
    <th>Layer</th>
    <th>Description</th>    
  </tr>
  <tr>
    <td>Input</td>
    <td>160x320x3</td>    
  </tr>
  <tr>
    <td>Convolution</td>
    <td>1x1 stride,padding="VALID",kernel = 5x5</td>   
  </tr>
  <tr>
    <td>MaxPooling2D</td>
    <td>Maxed pooled output</td>   
  </tr>
  <tr>
    <td>Convolution</td>
    <td>1x1 stride,padding="VALID",kernel = 5x5</td>   
  </tr>
  <tr>
	<td>MaxPooling2D</td>
    <td>Max Pooled output</td>
  </tr>
  <tr>
    <td>RELU</td>
    <td>Activated output</td>   
  </tr>
  <tr>
    <td>Dropout</td>
    <td>Use dropout with a factor of 0.3</td>   
  </tr>
  <tr>
    <td>Flatten</td>
    <td>Flatten the output for fully connected layer</td>   
  </tr>
  <tr>
    <td>FullyConnected</td>
    <td>output = 128</td>   
  </tr>
  <tr>
    <td>RELU</td>
    <td>output = 128</td>   
  </tr>
  <tr>
    <td>FullyConnected</td>
    <td>input = 128, output = 60</td>   
  </tr>
  <tr>
    <td>RELU</td>
    <td>output = 60</td>   
  </tr>
  <tr>
    <td>FullyConnected</td>
    <td>input = 60, output = 1</td>   
  </tr>
  <tr>
    <td>AdamOptimizer</td>
    <td>Minimises the mean square error</td>   
  </tr>
</table>

 Once the model is created, it is run with 7 epochs and the output is saved as model.h5 file. The data is also shuffled to get a uniform distribution of the input images.
 
#### <b> Validate the model architecture on the validation set </b>

The model is than tested on the validation set. Initially I saw that the model performed well on the testing set but not on the validation set. A possible reason for this could be overfitting. I experimented with different values and saw that a value of 0.3 for dropout gave a very reasonable accuracy for 
validation loss. 20% of the training data has been used as validation data.


#### <b> Test the output of the model architecture on the simulator in autonomous mode. </b>
After the model is trained, the generated model.h file is run on the simulator with the help of drive.py to get the simulator to run in the
autonomous mode. It was initially observed that the car moved very slowly. Hence the data was augmented by flipping images and the test images were also
cropped so that the images contained only the relevant information needed to train the network. Modifications were also done to the network by
experimenting with different filters and kernel size as well as the number of convolutional layers.


#### <b>Record the video of the car in autonomous mode </b>
The images of the car running in autonomous mode is captured in the folder automode by running drive.py with the model.h5 file that is generated by
training the network. Once the images have been obtained, a video is created by combining these files using video.py file provided in the project
resources. The name of the file is automode.mp4 located in the root folder of the project.



