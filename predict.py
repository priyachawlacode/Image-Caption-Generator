# final prediction file , can be used straight forward for prediction of captions for any passed image.

import cv2
import numpy as np
from glob import glob

import tensorflow as tf
from tensorflow.keras.models import Model

from keras.models import model_from_json
from pickle import load
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences


def predict(path):
        # loading saved ResNet50.h5 model in current dir
        resnet50 = tf.keras.models.load_model('ResNet50.h5')

        # Check its architecture
        resnet50.summary()

        # -------------------------------------------

        # function for preparing test set for testing
        def getImage():
            
            test_img_path = images_path

            test_img = cv2.imread(test_img_path)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

            test_img = cv2.resize(test_img, (224,224))

            test_img = np.reshape(test_img, (1,224,224,3))
            
            return test_img

        # --------------------------------------------------------

        # load json and create model
        #json_file = open('C:\\Users\\msn21\\Desktop\\Minor Project\\new\\DESCIT\\DESCIT\\Predict\\model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #model = model_from_json(loaded_model_json)
        model = tf.keras.models.load_model('Predict/trainedmodel1500.h5')
        # load weights into new model
        model.load_weights("Predict/mine_model_weights.h5")
        print("Loaded model from disk")

        # ------------------------------------------


        #fetching new_dict and inv_dict
        new_dict = load(open('Predict/new_dict15000.p','rb'))
        inv_dict = load(open('Predict/inv_dict1500.p','rb'))

        # providing test image path
        images_path = path
        #timg = glob(images_path+'*.jpg')

        # -----------------------------------------------------

        # final prediction
        #no = 4
        test_img_path = images_path

        test_img = cv2.imread(test_img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_feature = resnet50.predict(getImage()).reshape(1,2048)
        text_inp = ['startofseq']
        count = 0
        caption = ''
        MAX_LEN = 36
        while count <36:
            count +=1
            encoded = []
            for i in text_inp:
                encoded.append(new_dict[i])
            encoded = [encoded]
            encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)
            prediction = np.argmax(model.predict([test_feature, encoded]))
            sampled_word = inv_dict[prediction]
                
            if sampled_word == 'endofseq':
                break
            caption = caption + ' ' + sampled_word

            text_inp.append(sampled_word)
        #plt.imshow(test_img)
        #plt.imshow(test_img)
        #plt.xlabel(caption)
        print('All Done')
        return(caption)

