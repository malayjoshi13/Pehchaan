# stage 3

# This script is used to find faces in database which are similar to user's input image.

# How to run:
# "python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset MTCNN VGGFace euclidean"
# above "MTCNN" is used as detector and "VGGFace" as recognizer and "euclidean" as distance_metric
# or 
# "python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset" 
# above by default, "RetinaFace" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric

import os
from CompareFaces import build_recognition_model, compare_faces
from utils import CheckImageAdded, CreateRepresentation
import pickle
import pandas as pd
import sys

def find_similar_faces(user_img_path, db_path, face_recog_model ='ArcFace', distance_metric = 'euclidean', face_detection_model = 'RetinaFace'):
    
    all_representations = None

    if os.path.isdir(db_path) == True:
        for folder in os.listdir(db_path):
            face_recog_model_initialised = build_recognition_model(face_recog_model)

            representation_file_name = "representations_%s.pkl" % (face_recog_model)
            representation_file_name = representation_file_name.replace("-", "_").lower()

            #----------------------------------------------------------------------

            # check if representation file already exists or not
            checker_var = CheckImageAdded.check_image_added(db_path+"/"+folder)

            # if representation file already exists and no new images are added, then no need to create/update representation file again
            if os.path.exists(db_path+"/"+folder+"/"+representation_file_name) and checker_var==False: 
                print("Representation file ",representation_file_name, "already found")


            # else if representation file don't exists (first time running) or new images are added, then: 
            elif not os.path.exists(db_path+"/"+folder+"/"+representation_file_name) or checker_var==True:
                CreateRepresentation.create_representation_file(db_path, folder, face_recog_model_initialised, face_detection_model, representation_file_name)

            #---------------------------------------------------------------------------

            # after checking and creating representation file, we will load it
            f = open(db_path+"/"+folder+"/"+representation_file_name, 'rb')
            all_representations = pickle.load(f)

            # now we have variable "all_representations" having pair of database's image_path and correseponding representations,
            # we will just convert this pair into dataframe "df"
            df = pd.DataFrame(all_representations, columns = ["identity", "%s_representation" % (model_name)])

            # variable to store result
            resp_obj = []

            # find representation for user's input image
            target_representation = CreateRepresentation.represent(user_img_path, face_detection_model_initialised = face_recog_model_initialised, face_recog_model = face_recog_model)

            #------------------------------------------------- 

            # now we have variable "df" having representation of database images (aka "source_representation") and 
            # variable "target_representation" having represntation of target image.
            # We will now just check which image in database has similar representation to target image
            for index, instance in df.iterrows():
                source_representation = instance["%s_representation" % (model_name)]
            
                _, identified = compare_faces(target_representation, source_representation, distance_metric, model_name, detector_backend)
                if identified:
                    # will add path of database images having representation close to target image
                    # in "resp_obj" list
                    resp_obj.append(df["identity"][index])

            #-------------------------------------------------

        print("images in database similar to target image are:", resp_obj)
        return resp_obj

    else:
        raise ValueError("Passed db_path does not exist!")


# Main driver
if __name__ == "__main__":
    img1_path = sys.argv[1]
    database = sys.argv[2]

    try:
        detector_backend = sys.argv[3]
    except:
        detector_backend = "RetinaFace"

    try:
        model_name = sys.argv[4]
    except:
        model_name = "ArcFace"

    try:
        distance_metric = sys.argv[5]
    except:
        distance_metric = 'euclidean'        

    result = find_similar_faces(img1_path, database, detector_backend = detector_backend, model_name = model_name, distance_metric = distance_metric)