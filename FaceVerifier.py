# second stage

from tqdm import tqdm
import numpy as np
import utils.distance_fn as distance_fn
import utils.threshold_fn as threshold_fn

import recognizers.ArcFace as ArcFace
import recognizers.DeepID as DeepID
import recognizers.Facenet as Facenet
import recognizers.Facenet512 as Facenet512
import recognizers.FbDeepFace as FbDeepFace
import recognizers.OpenFace as OpenFace
import recognizers.VGGFace as VGGFace
import recognizers.SFace as SFace

from FaceDetector import find_faces
import sys

def find_input_shape(model):

	#face recognition models have different size of inputs

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	#----------------------

	if type(input_shape) == list: 
		input_shape = tuple(input_shape)

	return input_shape

#------------------------------------------------------------------------------------

def represent(img_path, model, detector_backend ='retinaface'):

	#decide input shape
	input_shape_x, input_shape_y = find_input_shape(model)

	#detect and align
	img = find_faces(img_path = img_path, detector_backend = detector_backend, target_size=(input_shape_y, input_shape_x))

	#represent
	embedding = model.predict(img)[0].tolist()

	return embedding

#--------------------------------------------------------------------------------------

def build_model(model_name = 'ArcFace'):
    global model_obj

    models = {
    'VGG-Face': VGGFace.loadModel,
    'SFace': SFace.loadModel,
    'OpenFace': OpenFace.loadModel,
    'Facenet': Facenet.loadModel,
    'Facenet512': Facenet512.loadModel,
    'DeepFace': FbDeepFace.loadModel,
    'DeepID': DeepID.loadModel,
    'ArcFace': ArcFace.loadModel    
    }

    model = models.get(model_name)
    if model:
        model = model()
    else:
        raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model

#-----------------------------------------------------------------------------------------------

def similarity_finder(img1_representation, img2_representation, distance_metric = 'euclidean', model_name = 'ArcFace', detector_backend = 'retinaface'):
   
   # find distances between embeddings
   # more distance, less similarity
   # thus distance must not be more than threshold value, else we consider it totally different

    if distance_metric == 'euclidean':
        distance = distance_fn.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = distance_fn.findEuclideanDistance(distance.l2_normalize(img1_representation), distance.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    distance = np.float64(distance) 


    #decision

    threshold = threshold_fn.findThreshold(model_name, distance_metric)

    if distance <= threshold:
        identified = True
        print("both faces match")
    else:
        identified = False
        print("both faces don't match")

    resp_obj = {
        "verified": identified
        , "distance": distance
        , "threshold": threshold
        , "model": model_name
        , "detector_backend": detector_backend
        , "similarity_metric": distance_metric
    }

    print(resp_obj)

    return resp_obj, identified

#-----------------------------------------------------------------------------------------

def verify(img1_path, img2_path, model_name = 'ArcFace', distance_metric = 'euclidean', detector_backend = 'retinaface'):

    img_list = [[img1_path, img2_path]]

    #--------------------------------

    model = build_model(model_name)

	#------------------------------

    pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = True)

    for index in pbar:

        instance = img_list[index]

        if type(instance) == list and len(instance) >= 2:
            img1_path = instance[0]; img2_path = instance[1]

            img1_representation = represent(img_path = img1_path, model = model
                    , detector_backend = detector_backend)

            img2_representation = represent(img_path = img2_path, model = model, 
                    detector_backend = detector_backend)

            similarity_finder(img1_representation, img2_representation, distance_metric, model_name, detector_backend)


# Main driver
if __name__ == "__main__":
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "SFace"]   
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    try:
        detector_backend = sys.argv[3]
    except:
        detector_backend = "retinaface"

    try:
        model_name = sys.argv[4]
    except:
        model_name = "ArcFace"

    try:
        distance_metric = sys.argv[5]
    except:
        distance_metric = 'euclidean'

    result = verify(img1_path, img2_path, detector_backend = detector_backend, model_name = model_name, distance_metric = distance_metric)

# "python FaceVerifier.py ./dataset/kalam/1.jpg ./dataset/kalam/hi.jpeg mtcnn VGG-Face euclidean"
# or "python FaceVerifier.py ./dataset/kalam/1.jpg ./dataset/kalam/hi.jpeg" as by default use 
#      "retinaface" as detector and "ArcFace" as recognizer and "euclidean" as distance_metric


