import os
import gdown
import zipfile
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout

def baseModel():
    base_model = Sequential()
    base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
    base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
    base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    base_model.add(Flatten(name='F0'))
    base_model.add(Dense(4096, activation='relu', name='F7'))
    base_model.add(Dropout(rate=0.5, name='D0'))
    base_model.add(Dense(8631, activation='softmax', name='F8'))
    return base_model 

def loadModel(url = 'https://drive.google.com/uc?id=1Gh-5OChUr4LKvELOpzyFUh_NHGvENX_F'): 

    model = baseModel()

    #-----------------------------------

    weight_location_in_local = os.path.join(os.getcwd(), 'recognizers', 'recognizers_weights', 'DeepFace', 'deepface_weights.h5')

    if not os.path.exists('./recognizers/recognizers_weights/DeepFace'):
        os.mkdir('./recognizers/recognizers_weights/DeepFace')	

    if os.path.isfile(weight_location_in_local) != True:
        print("deepface_weights.h5 will be downloaded...")		
        gdown.download(url, weight_location_in_local, quiet=False)

    #-----------------------------------

    model.load_weights(weight_location_in_local)	

    deepface = Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)
        
    return deepface