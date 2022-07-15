# stage 3

import os
from FaceVerifier import build_model, represent, similarity_finder
import pickle
from tqdm import tqdm
import pandas as pd
import sys

def NewImageAddedRemoved(path):
    answer = False # initiallly assume no new image is added in database
    count1 = 0
    count2 = 0

    # if tracker files are present, then first analyse them
    if os.path.isfile(path+'/files_tracker.txt') == True:
        # list of already existing images in database
        with open(path+'/files_tracker.txt') as f:
            lines = f.readline()

    if os.path.isfile(path+'/number_of_files.txt') == True:
        # number of already existing images in database
        with open(path+'/number_of_files.txt') as g:
            count1 = g.readline()
            count1 = int(count1)


        #------------------------------------------------

        # checking if any new image is added or not
        for file in os.listdir(path):
            file = path+"/"+file
            if ('.jpg' in file.lower()) or ('.png' in file.lower()) or ('.jpeg' in file.lower()):
                count2+=1
                if file not in lines:
                    print("new images detected in " + path+ " location. Recreating representations files")
                    # if found some new image in database, then set variable "answer" as True
                    answer = True

        #------------------------------------------------
        
        # checking if any image is deleted or not
        if count2<count1:
            answer = True

    # if tracker file don't exist, it clearly means that its first iteration, so here no new images would be added
    else:
        answer = False

    return answer




def find_similar_faces(img_path, db_path, model_name ='ArcFace', distance_metric = 'euclidean', detector_backend = 'retinaface'):
    all_representations = None
	#-------------------------------

    if os.path.isdir(db_path) == True:
        for folder in os.listdir(db_path):
            model = build_model(model_name)

            representation_file_name = "representations_%s.pkl" % (model_name)
            representation_file_name = representation_file_name.replace("-", "_").lower()

            #----------------------------------------------------------------------
            checker_var = NewImageAddedRemoved(db_path+"/"+folder)
            #----------------------------------------------------------------------

            # if representations already exists, and no new images are added
            #   then no need to create representations again

            if os.path.exists(db_path+"/"+folder+"/"+representation_file_name) and checker_var==False: 

                f = open(db_path+"/"+folder+"/"+representation_file_name, 'rb')
                all_representations = pickle.load(f)

                print("There are ", len(all_representations)," representations found in ",representation_file_name)

            #----------------------------------------------------------------------

            # else if representations file dont exists, or new images are added
            #    then create representation files from scratch

            elif not os.path.exists(db_path+"/"+folder+"/"+representation_file_name) or checker_var==True: 
                faces_list = []
                count = 0

                for file in os.listdir(db_path+"/"+folder):
                        if ('.jpg' in file.lower()) or ('.png' in file.lower()) or ('.jpeg' in file.lower()):
                            count+=1
                            exact_path = db_path+"/"+folder + "/" + file
                            faces_list.append(exact_path)

                with open(db_path+'/'+folder+'/files_tracker.txt', 'w') as f:
                    f.write(str(faces_list))

                with open(db_path+'/'+folder+'/number_of_files.txt', 'w') as f:
                    f.write(str(count))

                if len(faces_list) == 0:
                    raise ValueError("There is no image in ", db_path+"/"+folder," folder! Validate .jpg or .png files exist in this path.")

                # and store all new representations for db images there

                all_representations = []

                pbar = tqdm(range(0,len(faces_list)), desc='Finding representations', disable = True)

                for index in pbar:
                    face = faces_list[index]

                    instance = []
                    instance.append(face)

                    representation = represent(img_path = face, model = model, detector_backend = detector_backend)

                    instance.append(representation)

                    all_representations.append(instance)

                f = open(db_path+"/"+folder+"/"+representation_file_name, "wb")
                pickle.dump(all_representations, f)
                f.close()

                print("New representations stored in ",db_path+"/"+folder+"/"+representation_file_name," file.")
            # now, we got representations for new images in database

            #---------------------------------------------------------------------------

            # now we have variable "all_representations" having pair of image_path - its representations
            # that it got either from pre-existing pickled represenations file or newly created represntations
            #       we will just convert this pair into dataframe "df"
            df = pd.DataFrame(all_representations, columns = ["identity", "%s_representation" % (model_name)])

            resp_obj = []

            #------------------------------------------------- 

            # find representation for passed image
            target_representation = represent(img_path = img_path, model = model, detector_backend = detector_backend)

            #------------------------------------------------- 

            # now we have "df" having representation of database images (aka "source_representation") and 
            # "target_representation" having represntation of target image
            #       we will now just check which image in database has similar representation to target image
            for index, instance in df.iterrows():
                source_representation = instance["%s_representation" % (model_name)]
            
                _, identified = similarity_finder(target_representation, source_representation, distance_metric, model_name, detector_backend)
                if identified:
                    # will add path of database images having representation close to target image
                    # in "resp_obj" list
                    resp_obj.append(df["identity"][index])

                #----------------------------------

        print("images in database similar to target image are:", resp_obj)
        return resp_obj

    else:
        raise ValueError("Passed db_path does not exist!")


# Main driver
if __name__ == "__main__":
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "SFace"]   
    img1_path = sys.argv[1]
    database = sys.argv[2]

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

    result = find_similar_faces(img1_path, database, detector_backend = detector_backend, model_name = model_name, distance_metric = distance_metric)

    # "python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean"
    # or "python SimilarFaceFinder.py ./dataset/kalam/1.jpg ./dataset" as by default use 
    #       "retinaface" as detector and "ArcFace" as recognizer and "euclidean" as distance_metric