import os
import gdown
import zipfile
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout

#-------------------------------------
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

def loadModel(url = 'https://drive.google.com/uc?id=1aYbpMuaWCl7Bv-lxjZWe_FdUlKVNsEpB'):

    model = baseModel()

    weight_location_in_local = os.path.join(os.getcwd(), 'weights', 'DeepFace', 'VGGFace2_DeepFace_weights_val-0.9034.h5')

    if not os.path.exists('./weights/DeepFace'):
        os.mkdir('./weights/DeepFace')	

    if os.path.isfile(weight_location_in_local) != True:
        print("VGGFace2_DeepFace_weights.h5 will be downloaded...")		
        gdown.download(url, weight_location_in_local, quiet=False)
        
        with zipfile.ZipFile(weight_location_in_local, 'r') as zip_ref:
            zip_ref.extractall('./weights/DeepFace')
        
    model.load_weights(weight_location_in_local)	

    deepface = Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)
        
    return deepface