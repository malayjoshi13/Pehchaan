# stage 2

# This script is for verifying if two input images different or same.

# This script is used as a module in FaceRecongnizer.py and SimilarFaceFinder.py scripts

import numpy as np
from utils import distance_fn, threshold_fn
from recognizers import ArcFace, DeepID, Facenet, FbDeepFace, OpenFace, VGGFace, SFace

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")

def build_recognition_model(face_recog_model = 'ArcFace'):

    models = {
    'VGGFace': VGGFace.loadModel,
    'SFace': SFace.loadModel,
    'OpenFace': OpenFace.loadModel,
    'Facenet': Facenet.loadModel,
    'DeepFace': FbDeepFace.loadModel,
    'DeepID': DeepID.loadModel,
    'ArcFace': ArcFace.loadModel    
    }

    model = models.get(face_recog_model)
    if model:
        model = model()
    else:
        raise ValueError('Invalid model_name passed - {}'.format(face_recog_model))

    return model


def calculate_similarity(img1_representation, img2_representation, face_recog_model = 'ArcFace', distance_metric = 'euclidean', face_detection_model = 'RetinaFace'):
   
   # find distances between user's input image's embedding and database's image's embedding.
   # more the distance, less similarity.
   # if distance is more than threshold value, then we can say two images to be totally different.

    if distance_metric == 'euclidean':
        distance = distance_fn.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = distance_fn.findEuclideanDistance(distance.l2_normalize(img1_representation), distance.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    distance = np.float64(distance) 


    #decision

    threshold = threshold_fn.findThreshold(face_recog_model, distance_metric)

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
        , "face_recognition_model": face_recog_model
        , "face_detector_model": face_detection_model
        , "similarity_metric": distance_metric
    }

    print(resp_obj)

    return resp_obj, identified