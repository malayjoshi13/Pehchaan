import os
import numpy as np
import cv2 as cv
import gdown

class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)

class SFaceModel:

    def __init__(self, model_path):

        self.model = cv.FaceRecognizerSF.create(
            model = model_path,
            config = "",
            backend_id = 0,
            target_id = 0)

        self.layers = [_Layer()]

    def predict(self, image):
        # Preprocess
        input_blob = (image[0] * 255).astype(np.uint8)  # revert the image to original format and preprocess using the model

        # Forward
        embeddings = self.model.feature(input_blob)
        
        return embeddings


def loadModel(url = "https://drive.google.com/uc?id=10DHPe1qNJKU0bpDVE5YWH-UWDBbocDau"):  

    weight_location_in_local = os.path.join(os.getcwd(), 'weights', 'sface', 'sface_2021dec.onnx')

    if not os.path.exists('./weights/sface'):
        os.mkdir('./weights/sface')	

    if os.path.isfile(weight_location_in_local) != True:
        print("sface will be downloaded...")
        gdown.download(url, weight_location_in_local, quiet=False)

    #-----------------------------------

    sface = SFaceModel(model_path = weight_location_in_local)

    return sface