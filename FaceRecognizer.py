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



def recognize(img_path, db_path, face_detection_model ='ArcFace', distance_metric = 'euclidean', face_recog_model = 'retinaface'):
    results = find_similar_faces(img_path, db_path, face_recog_model = face_recog_model, face_detection_model = face_detection_model)
    label_list = list()
    for result in results:
        label = result.split('/')[-2]
        label_list.append(label)

    final_label = most_frequent(label_list)
    return final_label



# Main driver
if __name__ == "__main__":
    img1_path = sys.argv[1]
    database = sys.argv[2]

    try:
        face_recog_model = sys.argv[3]
    except:
        face_recog_model = "retinaface"

    try:
        face_detection_model = sys.argv[4]
    except:
        face_detection_model = "ArcFace"

    try:
        distance_metric = sys.argv[5]
    except:
        distance_metric = 'euclidean'        

    result = recognize(img1_path, database, face_recog_model = face_recog_model, face_detection_model = face_detection_model, distance_metric = distance_metric)
    print("identity of person in input image is:- ", result)


# "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGGFace euclidean" (here "mtcnn" is used as detector and "VGGFace" as recognizer and "euclidean" as distance_metric)
# or 
# "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset" (here by default, "retinaface" is used as detector and "ArcFace" as recognizer and "euclidean" as distance_metric)