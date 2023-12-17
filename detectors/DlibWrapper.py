import gdown
import bz2
import os
import dlib
 
def build_model():

    home = './detectors/detectors_weights/dlib/'

    if not os.path.exists(home):
        os.mkdir(home)

    weight1_location_in_local = './detectors/detectors_weights/dlib/shape_predictor_5_face_landmarks.dat'

    if os.path.isfile(weight1_location_in_local) != True:
        print("shape_predictor_5_face_landmarks.dat is going to be downloaded")
        url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
        output = url.split("/")[-1] # shape_predictor_5_face_landmarks.dat.bz2
        path = home + output # ./models/dlib/shape_predictor_5_face_landmarks.dat.bz2
        gdown.download(url, path, quiet=False)
        zipfile = bz2.BZ2File(path)
        data = zipfile.read()
        open(weight1_location_in_local, 'wb').write(data)

    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./detectors/detectors_weights/dlib/shape_predictor_5_face_landmarks.dat')

    detector = {}
    detector["face_detector"] = face_detector
    detector["sp"] = sp
    return detector


def detect_face(detector, img):
    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]

    sp = detector["sp"]
    face_detector = detector["face_detector"]
    detections = face_detector(img, 1)

    if len(detections) > 0:
        for idx, d in enumerate(detections):
            left = d.left(); right = d.right()
            top = d.top(); bottom = d.bottom()

            detected_face = img[max(0, top): min(bottom, img.shape[0]), max(0, left): min(right, img.shape[1])]

            img_region = [left, top, right - left, bottom - top]

            img_shape = sp(img, detections[idx])
            aligned_detected_face = dlib.get_face_chip(img, img_shape, size = detected_face.shape[0])

            resp.append((aligned_detected_face, img_region))
    return resp