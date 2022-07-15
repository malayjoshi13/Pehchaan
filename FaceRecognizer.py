# stage 4

from SimilarFaceFinder import find_similar_faces
import sys

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num



def recognize(img_path, db_path, model_name ='ArcFace', distance_metric = 'euclidean', detector_backend = 'retinaface'):
    results = find_similar_faces(img_path, db_path, detector_backend = detector_backend, model_name = model_name)
    label_list = list()
    for result in results:
        label = result.split('/')[-2]
        label_list.append(label)

    final_label = most_frequent(label_list)

    return final_label



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

    result = recognize(img1_path, database, detector_backend = detector_backend, model_name = model_name, distance_metric = distance_metric)
    print("identity of person in input image is:- ", result)
    # "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGG-Face euclidean"
    # or "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset" as by default use 
    #       "retinaface" as detector and "ArcFace" as recognizer and "euclidean" as distance_metric