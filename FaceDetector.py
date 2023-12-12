# finds face region out of the whole input image of a person

# If want to use it as a stanalone script, use:
# "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset mtcnn VGGFace euclidean"
# or 
# "python FaceRecognizer.py ./dataset/kalam/1.jpg ./dataset"

import numpy as np
import cv2
import detectors.OpenCvHaarCascadeWrapper as OpenCvHaarCascadeWrapper
import detectors.MtcnnWrapper as MtcnnWrapper
import detectors.OpenCvDNNWrapper as OpenCvDNNWrapper
import detectors.DlibWrapper as DlibWrapper
import detectors.RetinaFaceWrapper as RetinaFaceWrapper
import detectors.MediapipeWrapper as MediapipeWrapper
import sys
import warnings
warnings.filterwarnings("ignore")


def init_detector_build(detector_backend = 'retinaface'):

    # We create a dictionary having multiple options for initiating detectors
    build_model_options = {
        'opencvhaar': OpenCvHaarCascadeWrapper.build_model,
        'opencvdnn': OpenCvDNNWrapper.build_model,
        'mtcnn': MtcnnWrapper.build_model,
        'dlib': DlibWrapper.build_model,
        'retinaface': RetinaFaceWrapper.build_model,
	    'mediapipe': MediapipeWrapper.build_model        
    }

    # We choose "initiation option" cooresponding to our detector from "build_model_options"
    #   dictionary and save it in "face_detector_builder" variable   
    face_detector_builder = build_model_options.get(detector_backend)
    # If "initiation option" choosed is valid, 
    if face_detector_builder:
        # then we call this function "face_detector_builder", i.e actually calling 
        # lets say "OpenCvWrapper.build_model" function (value from dictionary corresponding
        # to our selected detector)
        face_detector_builder_value = face_detector_builder()
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_builder_value



def init_detector_detection(face_detector, img, detector_backend = 'retinaface'):

    # We create a dictionary having multiple options for starting prediction work for 
    #   different detectors
    prediction_from_model_options = {
        'opencvhaar': OpenCvHaarCascadeWrapper.detect_face,
        'opencvdnn': OpenCvDNNWrapper.detect_face,
        'mtcnn': MtcnnWrapper.detect_face,
        'dlib': DlibWrapper.detect_face,
        'retinaface': RetinaFaceWrapper.detect_face,
	    'mediapipe': MediapipeWrapper.detect_face        
    }

    # We choose "prediction option" cooresponding to our detector from "prediction_from_model_options"
    #   dictionary and save it in "face_detector_predictor" variable 
    face_detector_predictor = prediction_from_model_options.get(detector_backend)

    # If "prediction option" choosed is valid,
    if face_detector_predictor:
        # then we call this function "face_detector_predictor", i.e actually calling 
        # lets say "OpenCvWrapper.detect_face" function (value from dictionary corresponding
        # to our selected detector)
        # Output of this function, i.e list of detected_face and region pair
        #   gets saved in "obj" variable
        obj = face_detector_predictor(face_detector, img)
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)

    #..............................................

    if len(obj) > 0:
        face, region = obj[0] #discard multiple faces
    else: #len(obj) == 0
        face = None
        region = [0, 0, img.shape[0], img.shape[1]]

    return face, region



def find_faces(img_path, detector_backend = 'retinaface', target_size = (224, 224)):

    # Read the image on which face detection is to be performed
    img = cv2.imread(img_path)
    base_img = img.copy()

    #--------------------------

    # Initiate face detection model
    face_detector = init_detector_build(detector_backend)

    # Perform face detection on the image using "detect_face" function.
    # This function outputs "detected_face" which has pixels of detected part of face
    # and "img_region" which has coordinates of detected part of face 
    # Right now we will use "detected_face" output of this function
        # If this function could detect face, it will store pixels of 
        #    detected face in "detected_face" variable
        # If it cannot detect face, it will store "None" in "detected_face" variable
    try:
        detected_face, img_region = init_detector_detection(face_detector, img, detector_backend)
        print("face found in this image")
    except: 
        detected_face = None
        print("no face found in this image")

    # If "detected_face" is not Nill but an array of pixels of detected part of face,
    #     then these pixels gets stored in "img" variable
    # But if "detected_face" is Nill (as no face was detected),
    #     then pixels of raw image will be stored in "img" variable
    if (isinstance(detected_face, np.ndarray)):
        img = detected_face
    elif detected_face == None:
        img = base_img

	#---------------------------------------------------
	
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

	#---------------------------------------------------

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

    return img_pixels


# Main driver
if __name__ == "__main__":
     backends = ['opencvhaar', 'opencvdnn', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
     img_path = sys.argv[1]
     try:
        detector_backend = sys.argv[2]
     except:
         detector_backend = 'retinaface'
     face = find_faces(img_path, detector_backend, target_size = (224, 224))

# just run "python FaceDetector.py ./dataset/kalam/1.jpg mtcnn"
# or "python FaceDetector.py ./dataset/kalam/1.jpg" as by default it has "retinaface"