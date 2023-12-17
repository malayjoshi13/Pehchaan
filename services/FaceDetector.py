# stage 1

# This script finds face region out of the whole input image of a person

# This script is used as a module in "utils/CreateRepresentation.py" script which is in-return used by "FaceRecongnizer.py" script

import numpy as np
import cv2
# from detectors import OpenCvHaarCascadeWrapper, MtcnnWrapper, OpenCvDNNWrapper, DlibWrapper, RetinaFaceWrapper, MediapipeWrapper 
from detectors import OpenCvHaarCascadeWrapper, MtcnnWrapper, OpenCvDNNWrapper, RetinaFaceWrapper, MediapipeWrapper 

import warnings
warnings.filterwarnings("ignore")

def build_detector_model(face_detection_model = 'RetinaFace'):

    # We create a dictionary having multiple options for initiating detectors
    build_model_options = {
        'OpenCVHaar': OpenCvHaarCascadeWrapper.build_model,
        'OpenCVDNN': OpenCvDNNWrapper.build_model,
        'MTCNN': MtcnnWrapper.build_model,
        # 'dlib': DlibWrapper.build_model,
        'RetinaFace': RetinaFaceWrapper.build_model,
	    'MediaPipe': MediapipeWrapper.build_model        
    }

    face_detector_builder = build_model_options.get(face_detection_model)
    if face_detector_builder:
        face_detector_builded = face_detector_builder()
    else:
        raise ValueError("invalid detector_backend passed - " + face_detection_model)

    return face_detector_builded


def initiate_detection(face_detector, img, face_detection_model = 'RetinaFace'):

    # We create a dictionary having multiple options for starting prediction work for different detectors
    prediction_from_model_options = {
        'OpenCVHaar': OpenCvHaarCascadeWrapper.detect_face,
        'OpenCVDNN': OpenCvDNNWrapper.detect_face,
        'MTCNN': MtcnnWrapper.detect_face,
        # 'dlib': DlibWrapper.detect_face,
        'RetinaFace': RetinaFaceWrapper.detect_face,
	    'MediaPipe': MediapipeWrapper.detect_face        
    }

    # We choose "prediction option" cooresponding to our detector from "prediction_from_model_options"
    # dictionary and save it in "detection_initiaited" variable 
    detection_initiaited = prediction_from_model_options.get(face_detection_model)

    if detection_initiaited:
        # then we call this function "face_detector_predictor", i.e actually calling 
        # lets say "OpenCvWrapper.detect_face" function (value from dictionary corresponding
        # to our selected detector)
        # Output of this function, i.e list of detected_face and region pair
        #   gets saved in "obj" variable
        obj = detection_initiaited(face_detector, img)
    else:
        raise ValueError("invalid detector_backend passed - " + face_detection_model)

    #..............................................

    if len(obj) > 0:
        face, region = obj[0] #discard multiple faces
    else: #len(obj) == 0
        face = None
        region = [0, 0, img.shape[0], img.shape[1]]

    return face, region


def find_faces(img_path, face_detection_model = 'RetinaFace', target_size = (224, 224)):

    img = cv2.imread(img_path)
    base_img = img.copy()

    # Initiate face detection model
    face_detector_builded = build_detector_model(face_detection_model)

    # Perform face detection on the image.
    # This function outputs "detected_face" which has pixels of detected part of face
    # and "img_region" which has coordinates of detected part of face 
    # Right now we will use "detected_face" output of this function:
        # If this function could detect face, it will store pixels of detected face in "detected_face" variable
        # If it cannot detect face, it will store "None" in "detected_face" variable
    try:
        detected_face, img_region = initiate_detection(face_detector_builded, img, face_detection_model)
        result = "Face found in this image"
    except: 
        detected_face = None
        result = "No face found in this image"


    # If "detected_face" is not Nill but an array of pixels of detected part of face,
    #     then these pixels gets stored in "img" variable
    # But if "detected_face" is Nill (as no face was detected),
    #     then pixels of raw image will be stored in "img" variable
    if (isinstance(detected_face, np.ndarray)):
        img = detected_face
    elif detected_face == None:
        img = base_img


    # Resize image to expected shape
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

	# Normalizing the image pixels for passing ahead to face recognition task
    img_pixels = np.asarray(img)
    imgpixels_to_img = img_pixels.copy()

    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels = img_pixels/255

	#---------------------------------------------------

    # For saving detected face, if you want

    # imgpixels_to_img = imgpixels_to_img[:, :, ::-1] #bgr to rgb
    # data = Image.fromarray(imgpixels_to_img, 'RGB')
    # data.save('dataset/kalam/5.jpg')
    # data.show()

    return result, img_pixels