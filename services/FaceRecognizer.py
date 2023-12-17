# stage 4

# This script is used to get label of user's input image.
 
# How to run as a standalone script:
# "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset MTCNN VGGFace euclidean" 
# above "MTCNN" is used as detector and "VGGFace" as recognizer and "euclidean" as distance_metric
# or 
# "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset" 
# above by default, "RetinaFace" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric

import sys
import services.SimilarFaceFinder as SimilarFaceFinder

import warnings
# To ignore all warnings
warnings.filterwarnings("ignore")

def most_frequent(List):
    counter = 0
    num = ""
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i 
    return num

def recognize(img_path, db_path, face_recog_model ='ArcFace', distance_metric = 'euclidean', face_detection_model = 'RetinaFace'):    

    label_list = list()
    
    results = SimilarFaceFinder.find_similar_faces(img_path, db_path, face_recog_model = face_recog_model, distance_metric = distance_metric, face_detection_model = face_detection_model)

    print("")
    print("")
    print("------------------------------------------------------")
    print("")
    print("")

    print("images in database similar to target image are:", results)    
    
    for result in results:
        label = result.split('/')[-2]
        label_list.append(label)

    print(label_list)
    final_label = most_frequent(label_list)
    return final_label

# # Main driver
# if __name__ == "__main__":
#     img1_path = sys.argv[1]
#     database = sys.argv[2]

#     try:
#         face_detection_model = sys.argv[3]
#     except:
#         face_detection_model = "RetinaFace"

#     try:
#         face_recog_model = sys.argv[4]
#     except:
#         face_recog_model = "ArcFace"

#     try:
#         distance_metric = sys.argv[5]
#     except:
#         distance_metric = 'euclidean'        

#     result = recognize(img1_path, database, face_recog_model = face_recog_model, distance_metric = distance_metric, face_detection_model = face_detection_model)
#     print("identity of person in input image is:-", result)