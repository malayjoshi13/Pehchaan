import recognizers.Facenet as Facenet
import os
import gdown

def loadModel(url = 'https://drive.google.com/uc?id=1wn1GWbVt5EbZKD7J0Cw6ky7X1Dz8A5pw'):
    model = Facenet.InceptionResNetV2(dimension = 512)
    
    #-----------------------------------

    weight_location_in_local = os.path.join(os.getcwd(), 'weights', 'facenet512', 'facenet512_weights.h5')

    if not os.path.exists('./weights/facenet512'):
        os.mkdir('./weights/facenet512')	

    if os.path.isfile(weight_location_in_local) != True:
        print("facenet512_weights.h5 will be downloaded...")
        gdown.download(url, weight_location_in_local, quiet=False)

    #-----------------------------------

    model.load_weights(weight_location_in_local)

    #-------------------------

    return model

