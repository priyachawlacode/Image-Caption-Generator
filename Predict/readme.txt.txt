This Read Me is contains all the instructions for predicting

Steps 1: Import libraries - cv2, glob, pickle, matplotlib, keras 2 libs..

Step 2: Select and download the model - Our Case ResNet50

Step 3: Load the files needed
A) new_dict1500.p
B) inv_dict1500.p

Step 4: Provide path for image to be predicted

Step 5: Load the model using json and give its weight

Step 6: For the image provided use the model to predict.

We had an accuracy of 74.32% in the final model - So the results are 
not that best.

This happened because of hardware limitation, we were only able to train
the model for 1500 images only out of 6000 expected for training.

** By msn21 **