# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


This assignment is relatively easy. But, problems that are derived by the assignment need more concern. We introduce the assignment and attach the concern. 

To clone a human driver, you should find a good driver firstly. I have tried and found I am not. So, I invited a friend who was good at video games. I bought a gamepad for her, and she generated driving data for me. After one forward trip, she turned the car around and recorded one backward trip. In this simulated environment, it shows that the data of two trips is enough. The first concern is that who owns the data if we are building a real self-driving car, and how much the payment should be considered an even bargain. The driving skill is certainly not as cheap as a gamepad.

Now, we reprocess the data. I have not attached the data since the right of attribution is not clean. The forward trip is attached into ‘Left.rar’, and the backward trip is in ‘Right.rar’. Copy an executable ‘unrar.exe’ from the ‘WinRAR’ directory; run ‘ReadData.py’ to transform simulator generated files to Python Pickle files. At later, ‘PretreatmentData.py’ augment data by flip the figure. The experiment shows the additional data will lead a better CNN.

The last step is to train the model with ‘Train.py’. At this point, we encounter many lazy CNNs. We have opened this problem on ‘[Stackoverflow](https://stackoverflow.com/questions/47846824/how-to-prevent-a-lazy-convolutional-neural-network)’ and hope someone can solve it. Here is our CNN. I think the problem is related with the structure of the neural network. A model that can drive the car on the track is in directory ‘Model’. Use it with ‘drive.py’. 

![image](https://github.com/BlueBirdHouse/CarND-Behavioral-Cloning-P3/tree/master/Figs/Dmodel.png)