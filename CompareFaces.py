# stage 2

# This script is for verifying if two input images different or same.

# This script is used as a module in FaceRecongnizer.py and SimilarFaceFinder.py scripts

from tqdm import tqdm
import numpy as np
from utils import distance_fn, threshold_fn
from recognizers import ArcFace, DeepID, Facenet, Facenet512, FbDeepFace, OpenFace, VGGFace, SFace
from utils import CreateRepresentation

def build_recognition_model(face_recog_model = 'ArcFace'):

    models = {
    'VGGFace': VGGFace.loadModel,
    'SFace': SFace.loadModel,
    'OpenFace': OpenFace.loadModel,
    'Facenet': Facenet.loadModel,
    'Facenet512': Facenet512.loadModel,
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


def calculate_similarity(img1_representation, img2_representation, distance_metric = 'euclidean', face_recog_model = 'ArcFace', face_detection_model = 'RetinaFace'):
   
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


def compare_faces(img1_path, img2_path, face_recog_model = 'ArcFace', distance_metric = 'euclidean', face_detection_model = 'RetinaFace'):

    img_list = [[img1_path, img2_path]]

    face_recog_model_initialised = build_recognition_model(face_recog_model)

    pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = True)

    for index in pbar:

        instance = img_list[index]

        if type(instance) == list and len(instance) >= 2:
            img1_path = instance[0]; img2_path = instance[1]

            img1_representation = CreateRepresentation.represent(img_path = img1_path, face_recog_model_initialised = face_recog_model_initialised, face_detection_model = face_detection_model)

            img2_representation = CreateRepresentation.represent(img_path = img2_path, face_recog_model_initialised = face_recog_model_initialised, face_detection_model = face_detection_model)

            calculate_similarity(img1_representation, img2_representation, distance_metric, face_recog_model, face_detection_model)