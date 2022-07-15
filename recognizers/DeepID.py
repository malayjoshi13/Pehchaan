import os
import gdown
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout

#-------------------------------------
def baseModel():
    myInput = Input(shape=(55, 47, 3))

    x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(myInput)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
    x = Dropout(rate=0.99, name='D1')(x)

    x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
    x = Dropout(rate=0.99, name='D2')(x)

    x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
    x = Dropout(rate=0.99, name='D3')(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name = 'fc11')(x1)

    x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name = 'fc12')(x2)

    y = Add()([fc11, fc12])
    y = Activation('relu', name = 'deepid')(y)

    model = Model(inputs=[myInput], outputs=y)

    return model    



def loadModel(url = 'https://drive.google.com/uc?id=1nAyht9ErDYN9Hnn6rHUs-azJSZhCO_L6'):

	model = baseModel()

	#-----------------------------------
    
	weight_location_in_local = os.path.join(os.getcwd(), 'weights', 'deepid', 'deepid_keras_weights.h5')

	if not os.path.exists('./weights/deepid'):
		os.mkdir('./weights/deepid')	

	if os.path.isfile(weight_location_in_local) != True:
		print("deepid_keras_weights.h5 will be downloaded...")
		gdown.download(url, weight_location_in_local, quiet=False)

	#-----------------------------------
	
	model.load_weights(weight_location_in_local)

	return model