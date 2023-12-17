# This module create representation for each image,
# as well as if new images are added then it also creates a new representation file. 

from tqdm import tqdm
import os
import pickle
from utils import FindInputShape
import services.FaceDetector as FaceDetector

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")

# Create resprentation for each image
def represent(img_path, face_recog_model_initialised, face_detection_model ='RetinaFace'):

    # decide input shape cooresponding to face recognition model user has selected
    input_shape_x, input_shape_y = FindInputShape.find_input_shape(face_recog_model_initialised)

    # detect and align
    print("Finding face")
    result, img = FaceDetector.find_faces(img_path = img_path, face_detection_model = face_detection_model, target_size=(input_shape_x, input_shape_y))
    print(result)

    # generate represention 
    print("Creating representation")
    embedding = face_recog_model_initialised.predict(img)[0].tolist()
    
    return embedding


# Create a combined representation file for all images of a particular label
def create_representation_file(db_path, folder, face_recog_model_initialised, face_detection_model, representation_file_name):

    # 1. firstly update tracker file with all images in particular label's folder  
    faces_list = []

    for file in os.listdir(db_path+"/"+folder):
        if ('.jpg' in file.lower()) or ('.png' in file.lower()) or ('.jpeg' in file.lower()):
            exact_path = db_path+"/"+folder + "/" + file
            faces_list.append(exact_path)

    with open(db_path+'/'+folder+'/files_tracker.txt', 'w') as f:
        f.write(str(faces_list))

    if len(faces_list) == 0:
        raise ValueError("There is no image in ", db_path+"/"+folder," folder! Validate if .jpg or .png files exist in this path.")

    # 2. then store all representation for all images of a particular label
    all_representations = []

    pbar = tqdm(range(0,len(faces_list)), desc='Finding representations', disable = True)

    for index in pbar:
        face = faces_list[index]

        instance = []
        instance.append(face)

        representation = represent(img_path = face, face_recog_model_initialised = face_recog_model_initialised, face_detection_model = face_detection_model)

        instance.append(representation)

        all_representations.append(instance)

    f = open(db_path+"/"+folder+"/"+representation_file_name, "wb")
    pickle.dump(all_representations, f)
    f.close()

    print("New representations stored in ",db_path+"/"+folder+"/"+representation_file_name," file.")