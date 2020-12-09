Earlier we were downloading the ResNet50 model from tensorflow website every time. Now we have saved the model in saved_model folder with name my_model carrying the variables 
required for the model. Now rather than downloading we can directly import it from the saved model. Also relative changes are made in predict(1).py file for using that saved model.
The last run time for prediction through the Google Colab was 32s for 1st time and 20s for following times.

So, we have stepped forward in optimizing the speed.
