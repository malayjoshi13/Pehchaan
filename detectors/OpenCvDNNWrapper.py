import cv2
import numpy as np
import os
import gdown

def build_model():
    weight1_location_in_local = os.path.join(os.getcwd(), 'detectors', 'detectors_weights', 'opencvdnn', 'deploy.prototxt')
    weight2_location_in_local = os.path.join(os.getcwd(), 'detectors', 'detectors_weights', 'opencvdnn', 'res10_300x300_ssd_iter_140000.caffemodel')   
    
    if not os.path.exists('./detectors/detectors_weights/opencvdnn'):
        os.mkdir('./detectors/detectors_weights/opencvdnn')    

    if os.path.isfile(weight1_location_in_local) != True:
        try:
            url = 'https://raw.githubusercontent.com/mayank8200/Real-Time-Face-Detection/master/deploy.prototxt.txt' # the network definition
            print("deploy.prototxt will be downloaded from the url "+url)
            gdown.download(url, weight1_location_in_local, quiet=False)
        except:
            raise ValueError("Pre-trained weight could not be loaded!. You might try to download the pre-trained weights from the url "+ url
            + " and copy it to the ", weight1_location_in_local, "manually.")    
    
    if os.path.isfile(weight2_location_in_local) != True:
        try:
            url = 'https://raw.githubusercontent.com/mayank8200/Real-Time-Face-Detection/master/res10_300x300_ssd_iter_140000.caffemodel' # the learned weights
            print("deploy.prototxt will be downloaded from the url "+url)
            gdown.download(url, weight2_location_in_local, quiet=False)
        except:
            raise ValueError("Pre-trained weight could not be loaded!. You might try to download the pre-trained weights from the url "+ url
            + " and copy it to the ", weight2_location_in_local, "manually.") 

    print(weight1_location_in_local)
    print(weight2_location_in_local)
    face_detector = cv2.dnn.readNetFromCaffe(weight1_location_in_local, weight2_location_in_local)
    return face_detector


def detect_face(face_detector, img):
    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]

    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)  # "setInput" and "forward" are functions of opencv
    detections = face_detector.forward()
    
    # Loop over the detections, 
    for i in range(0, detections.shape[2]):
        # then extract the confidence (i.e., probability) associated with the prediction, 
        confidence = detections[0, 0, i, 2] # https://www.w3schools.com/python/numpy/numpy_array_indexing.asp
        # then filter out weak detections by ensuring higher `confidence`,
        if confidence > 0.5:
            # then compute the coordinates of the bounding box for the object
            # We multiplied x1, y1, x2, y2 coordinates (i.e 3 to 7 elements) with w nd h
            #       so that the coordinates are not according to 300, 300 image but according to
            #       image of original dimnesions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_face = img[startY:endY, startX:endX]
            img_region = [startX, startY, endX, endY]
            # opencvDNN has no function to align the face, so we skipped aligning for it
            resp.append((detected_face, img_region))

    return resp